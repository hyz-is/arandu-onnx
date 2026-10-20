package feature_test

import (
	"bytes"
	"errors"
	"slices"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/quantize"
	"github.com/hyz-is/arandu-onnx/rewrite"
)

// affineWithWeightInputs is affineModel as an IR 3 exporter writes it: w and
// b are initializers and graph inputs as well, at opset 11.
func affineWithWeightInputs(ir int64) []byte {
	g := message{}.
		sub(1, node("MatMul", []string{"x", "w"}, []string{"xw"})).
		sub(1, node("Add", []string{"xw", "b"}, []string{"y"})).
		text(2, "affine-inputs").
		sub(5, floatInitializer("w", []float32{1, 2, 3, 4, 5, 6}, 3, 2)).
		sub(5, floatInitializer("b", []float32{0.5, -0.5}, 2)).
		sub(11, valueInfo("x", protoFloat, -1, 3)).
		sub(11, valueInfo("w", protoFloat, 3, 2)).
		sub(11, valueInfo("b", protoFloat, 2)).
		sub(12, valueInfo("y", protoFloat, -1, 2))
	return modelAt(g, ir, 11)
}

// marshal encodes a model, failing the test on error.
func marshal(t *testing.T, m *graph.Model) []byte {
	t.Helper()
	out, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestInitializersOutOfInputsMakesTheWeightsConstants(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	body := affineWithWeightInputs(3)
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if m.Graph.Constant("w") != nil {
		t.Fatal("a weight listed as a graph input reads as a constant before the rewrite")
	}
	report, err := rewrite.InitializersOutOfInputs(m)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(report.Removed, []string{"w", "b"}) || report.InputsBefore != 3 || report.InputsAfter != 1 ||
		report.IRBefore != 3 || report.IRAfter != 4 {
		t.Fatalf("report = %+v", report)
	}
	if m.Graph.Constant("w") == nil || m.Graph.Constant("b") == nil || len(m.Graph.Initializers) != 2 {
		t.Fatal("the weights are not constants after the rewrite")
	}
	out := marshal(t, m)

	manifest := syntheticManifest("affine-inputs", "x", 2, "y", 2)
	manifest.Inputs[0].Shape = []int64{-1, 3}
	if diff := sameOnCPU(t, rt, manifest, manifest, affineWithWeightInputs(6), out, []int64{4, 3}); diff != 0 {
		t.Errorf("the rewritten model differs from the original by %g", diff)
	}
	// x = {1, 1, 1} gives {1+3+5+0.5, 2+4+6-0.5}, by hand.
	got := runOnce(t, rt, manifest, out, onnx.Tensor{Shape: []int64{1, 3}, Float32: []float32{1, 1, 1}})
	if !slices.Equal(got.Float32, []float32{9.5, 11.5}) {
		t.Errorf("y = %v, want [9.5 11.5]", got.Float32)
	}

	// Run again, nothing is left to take out.
	again, err := rewrite.InitializersOutOfInputs(m)
	if err != nil || len(again.Removed) != 0 || !bytes.Equal(marshal(t, m), out) {
		t.Errorf("a second run: %+v, %v", again, err)
	}
}

func TestInitializersOutOfInputsRefusesAGraphLeftWithoutInputs(t *testing.T) {
	g := message{}.
		sub(1, node("Relu", []string{"w"}, []string{"y"})).
		text(2, "only-weights").
		sub(5, floatInitializer("w", []float32{1, -1}, 2)).
		sub(11, valueInfo("w", protoFloat, 2)).
		sub(12, valueInfo("y", protoFloat, 2))
	body := modelAt(g, 3, 11)
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rewrite.InitializersOutOfInputs(m); !errors.Is(err, rewrite.ErrUnsupported) {
		t.Fatalf("InitializersOutOfInputs = %v, want ErrUnsupported", err)
	}
	if !bytes.Equal(marshal(t, m), body) {
		t.Error("a refused model was changed")
	}
}

func TestQDQQuantizesTheWeightsInitializersOutOfInputsFreed(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")

	// w1 listed as a graph input too, as an IR 3 exporter writes it.
	m, err := graph.Load(float)
	if err != nil {
		t.Fatal(err)
	}
	m.Graph.Inputs = append(m.Graph.Inputs, graph.NewValueInfo("w1", graph.TensorType{
		ElemType: graph.Float, Ranked: true, Shape: []graph.Dim{{Value: 4}, {Value: 3}, {Value: 3}, {Value: 3}},
	}))
	listed := marshal(t, m)

	_, report := export(t, listed, table, quantize.Options{})
	if len(report.Quantized) != 1 || len(report.Skipped) != 1 || !strings.Contains(report.Skipped[0].Reason, "InitializersOutOfInputs") {
		t.Fatalf("before the rewrite: quantized %v, skipped %v", report.Quantized, report.Skipped)
	}
	if _, err := rewrite.InitializersOutOfInputs(m); err != nil {
		t.Fatal(err)
	}
	exported, report := export(t, marshal(t, m), table, quantize.Options{})
	if len(report.Quantized) != 2 || len(report.Skipped) != 0 {
		t.Fatalf("after the rewrite: quantized %v, skipped %v", report.Quantized, report.Skipped)
	}
	assertTensorRTForm(t, loadGraph(t, exported))
}

func TestQDQQuantizesWeightsThatConstantNodesHold(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")

	// w1 as the value of a Constant node, as some exporters write weights.
	m, err := graph.Load(float)
	if err != nil {
		t.Fatal(err)
	}
	w1 := m.Graph.Initializer("w1")
	m.Graph.RemoveInitializer("w1")
	value := *w1
	value.Name = ""
	m.Graph.Nodes = append([]*graph.Node{{
		Name: "Constant_w1", OpType: "Constant", Outputs: []string{"w1"},
		Attributes: []*graph.Attribute{graph.TensorAttribute("value", &value)},
	}}, m.Graph.Nodes...)
	constants := marshal(t, m)

	exported, report := export(t, constants, table, quantize.Options{})
	if len(report.Quantized) != 2 || !slices.Contains(report.RemovedInitializers, "w1") {
		t.Fatalf("quantized %v, removed %v", report.Quantized, report.RemovedInitializers)
	}
	out, err := graph.Load(exported)
	if err != nil {
		t.Fatal(err)
	}
	if out.Graph.OpCounts()["Constant"] != 0 {
		t.Error("the Constant node of the float weight is left")
	}
	assertTensorRTForm(t, loadGraph(t, exported))
	worst, peak := compareOnCPU(t, rt, float, exported, "y")
	if worst > qdqTolerance*peak {
		t.Errorf("the export differs from the float model by %g of %g", worst, peak)
	}
}

// dropoutModel is x -> Dropout -> Relu -> y at an opset, with the Dropout
// inputs and outputs given, its mask a graph output when maskIsOutput.
func dropoutModel(opset int64, inputs, outputs []string, maskIsOutput bool, attributes []message, inits ...message) []byte {
	g := message{}.
		sub(1, node("Dropout", inputs, outputs, attributes...)).
		sub(1, node("Relu", []string{outputs[0]}, []string{"y"})).
		text(2, "dropout")
	for _, t := range inits {
		g = g.sub(5, t)
	}
	g = g.sub(11, valueInfo("x", protoFloat, -1, 4)).sub(12, valueInfo("y", protoFloat, -1, 4))
	if maskIsOutput {
		g = g.sub(12, valueInfo(outputs[1], protoBool, -1, 4))
	}
	return modelAt(g, 7, opset)
}

func TestDropoutToIdentityKeepsWhatInferenceComputes(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	manifest := syntheticManifest("dropout", "x", 2, "y", 2)

	for name, tc := range map[string]struct {
		body    []byte
		removed []string
	}{
		"opset 11, ratio attribute, mask unread": {
			body: dropoutModel(11, []string{"x"}, []string{"d", "mask"}, false, []message{floatAttribute("ratio", 0.4)}),
		},
		"opset 13, ratio and training_mode false": {
			body: dropoutModel(13, []string{"x", "ratio", "training"}, []string{"d"}, false, nil,
				floatInitializer("ratio", []float32{0.4}), boolInitializer("training", false)),
			removed: []string{"ratio", "training"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			m, err := graph.Load(tc.body)
			if err != nil {
				t.Fatal(err)
			}
			report, err := rewrite.DropoutToIdentity(m)
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Replaced) != 1 || !slices.Equal(report.Removed, tc.removed) {
				t.Fatalf("report = %+v", report)
			}
			counts := m.Graph.OpCounts()
			if counts["Dropout"] != 0 || counts["Identity"] != 1 || len(m.Graph.Initializers) != 0 {
				t.Fatalf("ops %v, %d initializers left", counts, len(m.Graph.Initializers))
			}
			out := marshal(t, m)
			if diff := sameOnCPU(t, rt, manifest, manifest, tc.body, out, []int64{3, 4}); diff != 0 {
				t.Errorf("the rewritten model differs from the original by %g", diff)
			}
		})
	}
}

func TestDropoutToIdentityRefusesADropoutThatDrops(t *testing.T) {
	for name, tc := range map[string]struct {
		body []byte
		want string
	}{
		"training mode on": {
			body: dropoutModel(13, []string{"x", "ratio", "training"}, []string{"d"}, false, nil,
				floatInitializer("ratio", []float32{0.4}), boolInitializer("training", true)),
			want: "training_mode is true",
		},
		"a mask that is read": {
			body: dropoutModel(13, []string{"x"}, []string{"d", "mask"}, true, nil),
			want: "mask output",
		},
		"a training mode computed at run time": {
			body: dropoutModel(13, []string{"x", "", "x"}, []string{"d"}, false, nil),
			want: "computed at run time",
		},
	} {
		t.Run(name, func(t *testing.T) {
			m, err := graph.Load(tc.body)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := rewrite.DropoutToIdentity(m); !errors.Is(err, rewrite.ErrUnsupported) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("DropoutToIdentity = %v, want ErrUnsupported naming %q", err, tc.want)
			}
			if !bytes.Equal(marshal(t, m), tc.body) {
				t.Error("a refused model was changed")
			}
		})
	}
}

// loadGraph decodes a model and returns its graph, failing the test on error.
func loadGraph(t *testing.T, body []byte) *graph.Graph {
	t.Helper()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	return m.Graph
}
