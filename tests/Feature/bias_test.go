package feature_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/quantize"
)

// optimizedByORT is the graph ONNX Runtime builds from a model at graph
// optimization level 1, the level every session gets before its nodes are
// given to the providers, TensorRT included.
func optimizedByORT(t *testing.T, body []byte) *graph.Graph {
	t.Helper()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	var inputs, outputs []string
	for _, v := range m.Graph.Inputs {
		if m.Graph.Initializer(v.Name) == nil {
			inputs = append(inputs, v.Name)
		}
	}
	for _, v := range m.Graph.Outputs {
		outputs = append(outputs, v.Name)
	}
	dir := t.TempDir()
	in, out := filepath.Join(dir, "model.onnx"), filepath.Join(dir, "optimized.onnx")
	if err := os.WriteFile(in, body, 0o600); err != nil {
		t.Fatal(err)
	}
	options, err := ort.NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = options.Destroy() }()
	if err := options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableBasic); err != nil {
		t.Fatal(err)
	}
	if err := options.SetOptimizedModelFilePath(out); err != nil {
		t.Fatal(err)
	}
	session, err := ort.NewDynamicAdvancedSession(in, inputs, outputs, options)
	if err != nil {
		t.Fatalf("ONNX Runtime refused the model: %v", err)
	}
	_ = session.Destroy()
	optimized, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	g, err := graph.Load(optimized)
	if err != nil {
		t.Fatal(err)
	}
	return g.Graph
}

// int32Dequantizations names the DequantizeLinear nodes of a graph that read
// an int32 constant, which TensorRT does not run.
func int32Dequantizations(g *graph.Graph) []string {
	var out []string
	for _, n := range g.Nodes {
		if n.OpType != "DequantizeLinear" {
			continue
		}
		if x := g.Constant(n.Input(0)); x != nil && x.DataType == graph.Int32 {
			out = append(out, n.Name)
		}
	}
	return out
}

// v030Form puts the biases QDQ detached back where v0.3.0 left them: a Conv
// with its one-dimensional bias, followed by nothing.
func v030Form(t *testing.T, body []byte) []byte {
	t.Helper()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	g := m.Graph
	for _, conv := range slices.Clone(g.Nodes) {
		if conv.OpType != "Conv" || conv.Input(2) != "" {
			continue
		}
		consumers := g.Consumers(conv.Outputs[0])
		if len(consumers) != 1 || consumers[0].OpType != "Add" {
			continue
		}
		add := consumers[0]
		broadcast := g.Initializer(add.Inputs[1])
		values, err := broadcast.Float32s()
		if err != nil {
			t.Fatal(err)
		}
		bias := graph.Float32Tensor(broadcast.Name+"_vector", []int64{int64(len(values))}, values)
		g.Initializers = append(g.Initializers, bias)
		conv.Inputs = append(conv.Inputs, bias.Name)
		conv.Outputs = add.Outputs
		g.Nodes = slices.DeleteFunc(g.Nodes, func(n *graph.Node) bool { return n == add })
	}
	out, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestQDQKeepsEveryBiasOutOfONNXRuntimesBiasQuantization(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")

	exported, report := export(t, float, table, quantize.Options{})
	// Conv_c1 -> Relu -> the QuantizeLinear of Conv_c2's input: the path ONNX
	// Runtime takes. Conv_c2 goes on to a Flatten and keeps its bias.
	if !slices.Equal(report.DetachedBiases, []string{"b1"}) {
		t.Fatalf("DetachedBiases = %v, want [b1]", report.DetachedBiases)
	}
	if err := quantize.CheckTensorRT(loadModel(t, exported)); err != nil {
		t.Fatalf("CheckTensorRT on the export: %v", err)
	}
	if got := int32Dequantizations(optimizedByORT(t, exported)); len(got) != 0 {
		t.Errorf("ONNX Runtime turned the biases of the export into int32 DequantizeLinear nodes: %v", got)
	}

	// The same export in the form v0.3.0 wrote: ONNX Runtime quantizes the
	// bias, which is what split a TensorRT graph on a node, and CheckTensorRT
	// now says so before a node does.
	old := v030Form(t, exported)
	if got := int32Dequantizations(optimizedByORT(t, old)); !slices.Equal(got, []string{"Conv_c1_bias_dq"}) {
		t.Errorf("ONNX Runtime's int32 DequantizeLinear nodes on the v0.3.0 form = %v, want [Conv_c1_bias_dq]", got)
	}
	err := quantize.CheckTensorRT(loadModel(t, old))
	if !errors.Is(err, quantize.ErrNotTensorRTQDQ) || !strings.Contains(err.Error(), "WeightBiasQuantization") || !strings.Contains(err.Error(), `"Conv_c1"`) {
		t.Errorf("CheckTensorRT on the v0.3.0 form = %v", err)
	}

	worst, peak := compareOnCPU(t, rt, float, exported, "y")
	if worst > qdqTolerance*peak {
		t.Errorf("the export differs from the float model by %g of %g", worst, peak)
	}
}

// mlpModel is x [N, 3] -> MatMul w1 -> Add b1 -> Relu -> MatMul w2 -> Add b2
// -> y [N, 2]: the bias of the first MatMul goes straight to the
// QuantizeLinear of the second once both are quantized.
func mlpModel() []byte {
	g := message{}.
		sub(1, node("MatMul", []string{"x", "w1"}, []string{"h"})).
		sub(1, node("Add", []string{"h", "b1"}, []string{"hb"})).
		sub(1, node("Relu", []string{"hb"}, []string{"r"})).
		sub(1, node("MatMul", []string{"r", "w2"}, []string{"o"})).
		sub(1, node("Add", []string{"o", "b2"}, []string{"y"})).
		text(2, "mlp").
		sub(5, floatInitializer("w1", pseudoRandom(21, 3*4, 0.5), 3, 4)).
		sub(5, floatInitializer("b1", pseudoRandom(22, 4, 0.1), 4)).
		sub(5, floatInitializer("w2", pseudoRandom(23, 4*2, 0.5), 4, 2)).
		sub(5, floatInitializer("b2", pseudoRandom(24, 2, 0.1), 2)).
		sub(11, valueInfo("x", protoFloat, -1, 3)).
		sub(12, valueInfo("y", protoFloat, -1, 2))
	return model(g)
}

func TestQDQKeepsAMatMulBiasOutOfONNXRuntimesBiasQuantization(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := mlpModel()
	manifest, dir := writeModel(t, syntheticManifest("mlp", "x", 2, "y", 2), float)
	var samples []map[string]onnx.Tensor
	for seed := range uint64(16) {
		samples = append(samples, map[string]onnx.Tensor{"x": randomTensor(700+seed, 4, 3)})
	}
	result, err := rt.Calibrate(context.Background(), manifest, dir, onnx.Samples(samples...), calibration.Options{})
	if err != nil {
		t.Fatal(err)
	}
	exported, report := export(t, float, result.Table, quantize.Options{Mode: quantize.ModeConvMatMul})
	if len(report.Quantized) != 2 || !slices.Equal(report.DetachedBiases, []string{"b1"}) {
		t.Fatalf("quantized %v, detached %v", report.Quantized, report.DetachedBiases)
	}
	if err := quantize.CheckTensorRT(loadModel(t, exported)); err != nil {
		t.Fatalf("CheckTensorRT: %v", err)
	}
	if got := int32Dequantizations(optimizedByORT(t, exported)); len(got) != 0 {
		t.Errorf("ONNX Runtime turned the MatMul bias into int32 DequantizeLinear nodes: %v", got)
	}
	// The v0.3.0 form, with the bias still [N]: ONNX Runtime fuses the MatMul
	// and the Add into a Gemm and quantizes its bias.
	old := loadModel(t, exported)
	b1 := graph.Float32Tensor("b1_vector", []int64{4}, pseudoRandom(22, 4, 0.1))
	old.Graph.Initializers = append(old.Graph.Initializers, b1)
	add := nodeNamed(old.Graph, "Add_hb")
	add.Inputs[1] = b1.Name
	oldBody, err := old.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if got := int32Dequantizations(optimizedByORT(t, oldBody)); len(got) == 0 {
		t.Error("ONNX Runtime left the [N] bias of the fused MatMul alone: the control proves nothing")
	}
	if err := quantize.CheckTensorRT(loadModel(t, oldBody)); !errors.Is(err, quantize.ErrNotTensorRTQDQ) || !strings.Contains(err.Error(), `MatMul "MatMul_h" with the bias Add "Add_hb"`) {
		t.Errorf("CheckTensorRT on the v0.3.0 form = %v", err)
	}

	m := syntheticManifest("mlp", "x", 2, "y", 2)
	if diff := sameOnCPU(t, rt, m, m, float, exported, []int64{4, 3}); diff > 0.05 {
		t.Errorf("the export differs from the float model by %g", diff)
	}
}

func TestCheckTensorRTFollowsADequantizedBiasAndAnInputItCannotDequantize(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")
	exported, _ := export(t, float, table, quantize.Options{Mode: quantize.ModeConvMatMul})

	for name, tc := range map[string]struct {
		change func(*graph.Graph)
		want   string
	}{
		"a bias dequantized and reshaped on its way to a Conv": {
			change: func(g *graph.Graph) {
				g.Initializers = append(g.Initializers,
					graph.Int8Tensor("b2_q", []int64{2}, []int8{3, -4}),
					graph.Float32Tensor("b2_s", nil, []float32{0.01}),
					graph.Int8Tensor("b2_z", nil, []int8{0}),
					graph.Int64Tensor("b2_shape", []int64{1}, []int64{2}))
				conv := nodeNamed(g, "Conv_c2")
				g.Nodes = append([]*graph.Node{
					{Name: "b2_dq", OpType: "DequantizeLinear", Inputs: []string{"b2_q", "b2_s", "b2_z"}, Outputs: []string{"b2_dqo"}},
					{Name: "b2_reshape", OpType: "Reshape", Inputs: []string{"b2_dqo", "b2_shape"}, Outputs: []string{"b2_r"}},
				}, g.Nodes...)
				conv.Inputs[2] = "b2_r"
			},
			want: `"b2_dq" feeds the bias of Conv "Conv_c2"`,
		},
		"a dequantized bias added after a matrix product": {
			change: func(g *graph.Graph) {
				g.Initializers = append(g.Initializers,
					graph.Int8Tensor("y_bias_q", []int64{5}, []int8{1, 2, 3, 4, 5}),
					graph.Float32Tensor("y_bias_s", nil, []float32{0.01}),
					graph.Int8Tensor("y_bias_z", nil, []int8{0}))
				gemm := nodeNamed(g, "Gemm_y")
				gemm.Inputs = gemm.Inputs[:2]
				gemm.Outputs = []string{"y_raw"}
				g.Nodes = append(g.Nodes,
					&graph.Node{Name: "y_bias_dq", OpType: "DequantizeLinear", Inputs: []string{"y_bias_q", "y_bias_s", "y_bias_z"}, Outputs: []string{"y_bias"}},
					&graph.Node{Name: "y_add", OpType: "Add", Inputs: []string{"y_raw", "y_bias"}, Outputs: []string{"y"}})
			},
			want: `"y_bias_dq" feeds the bias of the Add "y_add" after Gemm "Gemm_y"`,
		},
		"a DequantizeLinear of a float input": {
			change: func(g *graph.Graph) {
				g.Initializers = append(g.Initializers,
					graph.Float32Tensor("x_s", nil, []float32{0.01}), graph.Int8Tensor("x_z", nil, []int8{0}))
				g.Nodes = append([]*graph.Node{
					{Name: "x_dq", OpType: "DequantizeLinear", Inputs: []string{"x", "x_s", "x_z"}, Outputs: []string{"x_unused"}},
				}, g.Nodes...)
			},
			want: `"x_dq" reads "x", which is neither an int8 constant nor the output of a QuantizeLinear`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			m := loadModel(t, exported)
			tc.change(m.Graph)
			err := quantize.CheckTensorRT(m)
			if !errors.Is(err, quantize.ErrNotTensorRTQDQ) || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("CheckTensorRT = %v, want it to name %s", err, tc.want)
			}
		})
	}
}

// loadModel decodes a model, failing the test on error.
func loadModel(t *testing.T, body []byte) *graph.Model {
	t.Helper()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// nodeNamed returns the node of that name, or nil.
func nodeNamed(g *graph.Graph, name string) *graph.Node {
	for _, n := range g.Nodes {
		if n.Name == name {
			return n
		}
	}
	return nil
}
