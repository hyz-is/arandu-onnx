package feature_test

import (
	"bytes"
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/quantize"
)

// The QDQ tolerance. A float model and its INT8 export differ by the
// rounding of every quantized activation and weight; over this small network
// that is a few per cent of the output's range. The test holds the largest
// difference over calibration inputs to 5% of the largest |output|, which a
// wrong scale, axis or rounding exceeds many times over.
const qdqTolerance = 0.05

// pseudoRandom is n values uniform in [-scale, scale), the same for a seed.
func pseudoRandom(seed uint64, n int, scale float32) []float32 {
	random := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	out := make([]float32, n)
	for i := range out {
		out[i] = scale * (2*random.Float32() - 1)
	}
	return out
}

// convNet is x [1, 3, 8, 8] -> Conv(4, 3x3, pad 1) -> Relu -> Conv(2, 3x3,
// stride 2, pad 1) -> Flatten -> Gemm(32 -> 5) -> y [1, 5], with synthetic
// weights, at an opset and with extra nodes after y when given.
func convNet(opset int64, tail ...message) []byte {
	pads := intsAttribute("pads", 1, 1, 1, 1)
	g := message{}.
		sub(1, node("Conv", []string{"x", "w1", "b1"}, []string{"c1"}, intsAttribute("kernel_shape", 3, 3), pads)).
		sub(1, node("Relu", []string{"c1"}, []string{"r1"})).
		sub(1, node("Conv", []string{"r1", "w2", "b2"}, []string{"c2"}, intsAttribute("kernel_shape", 3, 3), pads, intsAttribute("strides", 2, 2))).
		sub(1, node("Flatten", []string{"c2"}, []string{"f"}, intAttribute("axis", 1))).
		sub(1, node("Gemm", []string{"f", "w3", "b3"}, []string{"y"}))
	output := "y"
	for _, n := range tail {
		g = g.sub(1, n)
		output = "z"
	}
	g = g.text(2, "convnet").
		sub(5, floatInitializer("w1", pseudoRandom(11, 4*3*3*3, 0.5), 4, 3, 3, 3)).
		sub(5, floatInitializer("b1", pseudoRandom(12, 4, 0.1), 4)).
		sub(5, floatInitializer("w2", pseudoRandom(13, 2*4*3*3, 0.5), 2, 4, 3, 3)).
		sub(5, floatInitializer("b2", pseudoRandom(14, 2, 0.1), 2)).
		sub(5, floatInitializer("w3", pseudoRandom(15, 32*5, 0.5), 32, 5)).
		sub(5, floatInitializer("b3", pseudoRandom(16, 5, 0.1), 5)).
		sub(11, valueInfo("x", protoFloat, 1, 3, 8, 8)).
		sub(12, valueInfo(output, protoFloat, 1, 5))
	ir := int64(7)
	if opset < 13 {
		ir = 6
	}
	return modelAt(g, ir, opset)
}

func convManifest(output string) onnx.Manifest {
	m := syntheticManifest("convnet", "x", 4, output, 2)
	m.Inputs[0].Shape = []int64{1, 3, 8, 8}
	return m
}

// calibrateConvNet calibrates the network on 32 synthetic inputs.
func calibrateConvNet(t *testing.T, rt *onnx.Runtime, body []byte, output string) calibration.Table {
	t.Helper()
	manifest, dir := writeModel(t, convManifest(output), body)
	var samples []map[string]onnx.Tensor
	for seed := range uint64(32) {
		samples = append(samples, map[string]onnx.Tensor{"x": randomTensor(100+seed, 1, 3, 8, 8)})
	}
	result, err := rt.Calibrate(context.Background(), manifest, dir, onnx.Samples(samples...), calibration.Options{})
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	return result.Table
}

// export runs QDQ on a copy of the model.
func export(t *testing.T, body []byte, table calibration.Table, opts quantize.Options) ([]byte, quantize.Report) {
	t.Helper()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	report, err := quantize.QDQ(m, table, opts)
	if err != nil {
		t.Fatalf("QDQ: %v", err)
	}
	out, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	return out, report
}

// compareOnCPU runs the float and the quantized model on eight of the
// calibration inputs and returns the largest difference and the largest
// |float output|. They are calibration inputs on purpose: a minmax range
// covers them, so what is measured is the export's rounding. An input whose
// activations exceed the calibrated ranges is clipped, which is the business
// of the calibration samples, not of the export: a held-out synthetic input
// takes this network's flattened tensor 40% past its calibrated range.
func compareOnCPU(t *testing.T, rt *onnx.Runtime, float, quantized []byte, output string) (float64, float64) {
	t.Helper()
	floatManifest := convManifest(output)
	int8Manifest := convManifest(output)
	int8Manifest.ID = "convnet.int8"
	int8Manifest.File = "convnet.int8.onnx"
	int8Manifest.Precision = onnx.PrecisionINT8
	worst, peak := 0.0, 0.0
	for seed := range uint64(8) {
		input := randomTensor(100+seed, 1, 3, 8, 8)
		want := runOnce(t, rt, floatManifest, float, input)
		got := runOnce(t, rt, int8Manifest, quantized, input)
		worst = math.Max(worst, maxAbsDiff(want, got))
		for _, v := range want.Float32 {
			peak = math.Max(peak, math.Abs(float64(v)))
		}
	}
	return worst, peak
}

func TestQDQExportsATensorRTQuantizationThatRunsCloseToTheFloatModel(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")

	for _, tc := range []struct {
		mode      quantize.Mode
		quantized []string
		weights   map[string]int
		removed   string
	}{
		{quantize.ModeConv, []string{"Conv_c1", "Conv_c2"}, map[string]int{"w1": 0, "w2": 0}, "w1,w2"},
		{quantize.ModeConvMatMul, []string{"Conv_c1", "Conv_c2", "Gemm_y"}, map[string]int{"w1": 0, "w2": 0, "w3": 1}, "w1,w2,w3"},
	} {
		t.Run(string(tc.mode), func(t *testing.T) {
			body, report := export(t, float, table, quantize.Options{Mode: tc.mode})

			var names []string
			for _, q := range report.Quantized {
				names = append(names, q.Node)
			}
			if strings.Join(names, ",") != strings.Join(tc.quantized, ",") {
				t.Errorf("Quantized = %v, want %v", names, tc.quantized)
			}
			if strings.Join(report.RemovedInitializers, ",") != tc.removed {
				t.Errorf("RemovedInitializers = %v, want %s", report.RemovedInitializers, tc.removed)
			}
			for _, w := range report.Weights {
				if axis, ok := tc.weights[w.Initializer]; !ok || axis != w.Axis {
					t.Errorf("weight %s quantized along axis %d, want %d", w.Initializer, w.Axis, axis)
				}
			}
			if report.OpsetBefore != 13 || report.OpsetAfter != 13 || len(report.ScalesSHA256) != 64 {
				t.Errorf("report = %+v", report)
			}
			for _, a := range report.Activations {
				if a.Scale != table[a.Tensor]/127 {
					t.Errorf("activation %s has scale %v, want range/127 = %v", a.Tensor, a.Scale, table[a.Tensor]/127)
				}
			}

			m, err := graph.Load(body)
			if err != nil {
				t.Fatal(err)
			}
			if err := quantize.CheckTensorRT(m); err != nil {
				t.Errorf("CheckTensorRT: %v", err)
			}
			assertTensorRTForm(t, m.Graph)

			_, again := export(t, float, table, quantize.Options{Mode: tc.mode})
			if again.ScalesSHA256 != report.ScalesSHA256 {
				t.Error("two exports of the same model and table have different scale digests")
			}

			worst, peak := compareOnCPU(t, rt, float, body, "y")
			t.Logf("%s: max |float - int8| = %.5f over 8 calibration inputs, max |float| = %.4f (%.2f%%)", tc.mode, worst, peak, 100*worst/peak)
			if worst > qdqTolerance*peak {
				t.Errorf("max |float - int8| = %v, more than %.0f%% of %v", worst, 100*qdqTolerance, peak)
			}

			// The control: the same export with the first weight's scales
			// doubled has to fail the tolerance, or the tolerance proves
			// nothing.
			scale := m.Graph.Initializer("w1_scale")
			scales, _ := scale.Float32s()
			for i := range scales {
				scales[i] *= 2
			}
			*scale = *graph.Float32Tensor(scale.Name, scale.Dims, scales)
			broken, _ := m.Marshal()
			if bad, _ := compareOnCPU(t, rt, float, broken, "y"); bad <= qdqTolerance*peak {
				t.Errorf("doubled scales differ by %v, within the tolerance %v: the tolerance is too loose", bad, qdqTolerance*peak)
			}
		})
	}
}

// assertTensorRTForm checks the export node by node, independently of
// CheckTensorRT: int8 zero points of 0, positive scales, biases read as the
// float initializers they were, and int8 weights.
func assertTensorRTForm(t *testing.T, g *graph.Graph) {
	t.Helper()
	produced := map[string]*graph.Node{}
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			produced[out] = n
		}
	}
	qdq := 0
	for _, n := range g.Nodes {
		switch n.OpType {
		case "QuantizeLinear", "DequantizeLinear":
			qdq++
			zp := g.Initializer(n.Input(2))
			if zp == nil || zp.DataType != graph.Int8 {
				t.Fatalf("%s %s has zero point %q, want an int8 initializer", n.OpType, n.Name, n.Input(2))
			}
			values, _ := zp.Int8s()
			for _, v := range values {
				if v != 0 {
					t.Errorf("%s %s has zero point %d", n.OpType, n.Name, v)
				}
			}
			scales, err := g.Initializer(n.Input(1)).Float32s()
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range scales {
				if !(s > 0) {
					t.Errorf("%s %s has scale %v", n.OpType, n.Name, s)
				}
			}
			if n.OpType == "DequantizeLinear" {
				if x := g.Initializer(n.Input(0)); x != nil && x.DataType != graph.Int8 {
					t.Errorf("DequantizeLinear %s reads %s %s", n.Name, x.DataType, x.Name)
				}
			}
		case "Conv", "Gemm":
			bias := n.Input(2)
			if p := produced[bias]; p != nil {
				t.Errorf("the bias of %s comes from %s %s, want the float initializer", n.Name, p.OpType, p.Name)
			}
			if b := g.Initializer(bias); b == nil || b.DataType != graph.Float {
				t.Errorf("the bias %q of %s is not a float initializer", bias, n.Name)
			}
			if p := produced[n.Input(1)]; p != nil && p.OpType == "DequantizeLinear" {
				if w := g.Initializer(p.Input(0)); w == nil || w.DataType != graph.Int8 || p.Attribute("axis") == nil {
					t.Errorf("the weight of %s is not an int8 initializer read per channel", n.Name)
				}
			}
		}
	}
	if qdq == 0 {
		t.Error("the export has no QuantizeLinear or DequantizeLinear")
	}
}

func TestQDQSkipsWhatItIsToldToAndWhatTheTableLeavesOut(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")

	_, byName := export(t, float, table, quantize.Options{ExcludeNodes: []string{"Conv_c2"}})
	if len(byName.Quantized) != 1 || byName.Quantized[0].Node != "Conv_c1" ||
		len(byName.Skipped) != 1 || byName.Skipped[0].Reason != "excluded by name" {
		t.Errorf("exclude by name: %+v", byName)
	}
	_, byPrefix := export(t, float, table, quantize.Options{Mode: quantize.ModeConvMatMul, ExcludePrefixes: []string{"Gemm_"}})
	if len(byPrefix.Quantized) != 2 || len(byPrefix.Skipped) != 1 || byPrefix.Skipped[0].Node != "Gemm_y" {
		t.Errorf("exclude by prefix: %+v", byPrefix)
	}

	partial := calibration.Table{}
	for name, r := range table {
		if name != "r1" {
			partial[name] = r
		}
	}
	body, missing := export(t, float, partial, quantize.Options{})
	if len(missing.Quantized) != 1 || len(missing.Skipped) != 1 || !strings.Contains(missing.Skipped[0].Reason, `"r1"`) {
		t.Errorf("a tensor the table leaves out: %+v", missing)
	}
	m, _ := graph.Load(body)
	if m.Graph.Initializer("w2") == nil {
		t.Error("the weight of the node left in float was removed")
	}
}

func TestQDQRaisesASafeOldOpsetAndRefusesAnUnsafeOne(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})

	old := convNet(11)
	table := calibrateConvNet(t, rt, old, "y")
	body, report := export(t, old, table, quantize.Options{})
	if report.OpsetBefore != 11 || report.OpsetAfter != 13 {
		t.Errorf("opset %d -> %d, want 11 -> 13", report.OpsetBefore, report.OpsetAfter)
	}
	m, _ := graph.Load(body)
	if v, _ := m.Opset(""); v != 13 || m.IRVersion != 7 {
		t.Errorf("written at opset %d, IR %d; want 13 and 7", v, m.IRVersion)
	}
	worst, peak := compareOnCPU(t, rt, old, body, "y")
	if worst > qdqTolerance*peak {
		t.Errorf("the raised model differs by %v of %v", worst, peak)
	}

	// Softmax at opset 11 coerces to 2-D; at 13 it works on one axis.
	withSoftmax := convNet(11, node("Softmax", []string{"y"}, []string{"z"}, intAttribute("axis", 1)))
	softmaxTable := calibrateConvNet(t, rt, withSoftmax, "z")
	sm, _ := graph.Load(withSoftmax)
	if _, err := quantize.QDQ(sm, softmaxTable, quantize.Options{}); !errors.Is(err, quantize.ErrRefused) || !strings.Contains(err.Error(), "Softmax") {
		t.Errorf("opset 11 with Softmax: %v, want ErrRefused naming Softmax", err)
	}
	if out, _ := sm.Marshal(); !bytes.Equal(out, withSoftmax) {
		t.Error("the refused model was changed")
	}

	ancient, _ := graph.Load(convNet(9))
	if _, err := quantize.QDQ(ancient, table, quantize.Options{}); !errors.Is(err, quantize.ErrRefused) {
		t.Errorf("opset 9: %v, want ErrRefused", err)
	}
}

func TestQDQRefusesWhatItCannotQuantize(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")
	quantized, _ := export(t, float, table, quantize.Options{})

	for name, tc := range map[string]struct {
		body  []byte
		table calibration.Table
		opts  quantize.Options
		want  string
	}{
		"already quantized":    {quantized, table, quantize.Options{}, "already quantized"},
		"another model":        {float, calibration.Table{"elsewhere": 1}, quantize.Options{}, "another model"},
		"an empty table":       {float, calibration.Table{}, quantize.Options{}, "no tensor"},
		"nothing left":         {float, table, quantize.Options{ExcludeOpTypes: []string{"Conv"}}, "no node is left"},
		"an unknown mode":      {float, table, quantize.Options{Mode: "everything"}, "mode"},
		"external weights":     {externalWeights(), table, quantize.Options{}, "external data"},
		"a non-positive range": {float, calibration.Table{"x": -1}, quantize.Options{}, "positive"},
	} {
		m, err := graph.Load(tc.body)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		_, err = quantize.QDQ(m, tc.table, tc.opts)
		if !errors.Is(err, quantize.ErrRefused) || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want ErrRefused saying %q", name, err, tc.want)
		}
		if out, _ := m.Marshal(); !bytes.Equal(out, tc.body) {
			t.Errorf("%s: the refused model was changed", name)
		}
	}
}

// externalWeights is one Conv whose weight w1 is in an external data file
// (TensorProto.data_location = 1, external_data = 13), as a large export
// writes its weights.
func externalWeights() []byte {
	location := message{}.text(1, "location").text(2, "weights.bin")
	w1 := message{}.integer(1, 4).integer(1, 3).integer(1, 3).integer(1, 3).integer(2, protoFloat).text(8, "w1").
		sub(13, location).integer(14, 1)
	pads := intsAttribute("pads", 1, 1, 1, 1)
	g := message{}.
		sub(1, node("Conv", []string{"x", "w1", "b1"}, []string{"y"}, intsAttribute("kernel_shape", 3, 3), pads)).
		text(2, "external").
		sub(5, w1).
		sub(5, floatInitializer("b1", pseudoRandom(12, 4, 0.1), 4)).
		sub(11, valueInfo("x", protoFloat, 1, 3, 8, 8)).
		sub(12, valueInfo("y", protoFloat, 1, 4, 8, 8))
	return modelAt(g, 7, 13)
}

func TestCheckTensorRTNamesWhatTensorRTWouldRefuse(t *testing.T) {
	t.Parallel()

	uint8ZeroPoint := message{}.integer(2, int64(protoUint8)).text(8, "zp").bytes(9, []byte{128})
	asymmetric := message{}.
		sub(1, node("QuantizeLinear", []string{"x", "s", "zp"}, []string{"q"})).
		sub(1, node("DequantizeLinear", []string{"q", "s", "zp"}, []string{"y"})).
		text(2, "asymmetric").
		sub(5, floatInitializer("s", []float32{0.1})).
		sub(5, uint8ZeroPoint).
		sub(11, valueInfo("x", protoFloat, 4)).
		sub(12, valueInfo("y", protoFloat, 4))

	int32Bias := message{}.integer(1, 4).integer(2, 6).text(8, "bq").bytes(9, make([]byte, 16))
	int8Zero := message{}.integer(2, 3).text(8, "z8").bytes(9, []byte{0})
	biasDQ := message{}.
		sub(1, node("DequantizeLinear", []string{"bq", "s", "z8"}, []string{"b"})).
		sub(1, node("Conv", []string{"x", "w", "b"}, []string{"y"}, intsAttribute("kernel_shape", 1, 1))).
		text(2, "bias").
		sub(5, floatInitializer("s", []float32{0.1})).
		sub(5, int32Bias).
		sub(5, int8Zero).
		sub(5, floatInitializer("w", pseudoRandom(1, 4*3, 1), 4, 3, 1, 1)).
		sub(11, valueInfo("x", protoFloat, 1, 3, 2, 2)).
		sub(12, valueInfo("y", protoFloat, 1, 4, 2, 2))

	for name, tc := range map[string]struct {
		body []byte
		want []string
	}{
		"asymmetric uint8":   {modelAt(asymmetric, 7, 13), []string{"uint8 zero point"}},
		"a dequantized bias": {modelAt(biasDQ, 7, 13), []string{"reads \"bq\", which is int32", "feeds the bias"}},
		"a float model":      {convNet(13), []string{"no DequantizeLinear"}},
		"QOperator": {modelAt(message{}.sub(1, node("QLinearConv", []string{"x"}, []string{"y"})).
			sub(11, valueInfo("x", protoUint8, 1)).sub(12, valueInfo("y", protoUint8, 1)), 7, 13), []string{"QOperator"}},
	} {
		m, err := graph.Load(tc.body)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		err = quantize.CheckTensorRT(m)
		if !errors.Is(err, quantize.ErrNotTensorRTQDQ) {
			t.Errorf("%s: %v, want ErrNotTensorRTQDQ", name, err)
			continue
		}
		for _, want := range tc.want {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("%s: %v, want it to say %q", name, err, want)
			}
		}
	}
}
