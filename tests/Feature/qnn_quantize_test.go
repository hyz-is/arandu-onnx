package feature_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/quantize"
)

// qnnNet is x [1, 3, 8, 8] -> Conv(4, 3x3, pad 1) -> Sigmoid -> Mul (a SiLU)
// -> MaxPool(2) -> Conv(2, 1x1) -> y [1, 2, 4, 4], with synthetic weights.
func qnnNet() []byte {
	g := message{}.
		sub(1, node("Conv", []string{"x", "w1", "b1"}, []string{"c1"}, intsAttribute("kernel_shape", 3, 3), intsAttribute("pads", 1, 1, 1, 1))).
		sub(1, node("Sigmoid", []string{"c1"}, []string{"s1"})).
		sub(1, node("Mul", []string{"c1", "s1"}, []string{"m1"})).
		sub(1, node("MaxPool", []string{"m1"}, []string{"p1"}, intsAttribute("kernel_shape", 2, 2), intsAttribute("strides", 2, 2))).
		sub(1, node("Conv", []string{"p1", "w2", "b2"}, []string{"y"}, intsAttribute("kernel_shape", 1, 1))).
		text(2, "qnnnet").
		sub(5, floatInitializer("w1", pseudoRandom(21, 4*3*3*3, 0.4), 4, 3, 3, 3)).
		sub(5, floatInitializer("b1", pseudoRandom(22, 4, 0.1), 4)).
		sub(5, floatInitializer("w2", pseudoRandom(23, 2*4, 0.5), 2, 4, 1, 1)).
		sub(5, floatInitializer("b2", pseudoRandom(24, 2, 0.1), 2)).
		sub(11, valueInfo("x", protoFloat, 1, 3, 8, 8)).
		sub(12, valueInfo("y", protoFloat, 1, 2, 4, 4))
	return modelAt(g, 8, 17)
}

// qnnRanges covers every tensor of qnnNet over inputs in [-1, 1].
func qnnRanges() map[string]quantize.Range {
	return map[string]quantize.Range{
		"x":  {Min: -1, Max: 1},
		"c1": {Min: -6, Max: 6},
		"m1": {Min: -0.3, Max: 6},
		"y":  {Min: -4, Max: 4},
	}
}

func qnnManifest(input onnx.ElementType) onnx.Manifest {
	m := syntheticManifest("qnnnet", "x", 4, "y", 4)
	m.Inputs[0].Shape = []int64{1, 3, 8, 8}
	m.Inputs[0].ElementType = input
	m.Outputs[0].Shape = []int64{1, 2, 4, 4}
	return m
}

func qnnExport(t *testing.T, opts quantize.QNNOptions) ([]byte, quantize.QNNReport) {
	t.Helper()
	m, err := graph.Load(qnnNet())
	if err != nil {
		t.Fatal(err)
	}
	report, err := quantize.QNN(m, opts)
	if err != nil {
		t.Fatalf("QNN: %v", err)
	}
	out, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	return out, report
}

func tensorParams(report quantize.QNNReport, name string) (quantize.QNNTensor, bool) {
	for _, t := range report.Tensors {
		if t.Tensor == name {
			return t, true
		}
	}
	return quantize.QNNTensor{}, false
}

func TestQNNWritesTheAsymmetricUnsignedFormWithTheScalesQNNRequires(t *testing.T) {
	body, report := qnnExport(t, quantize.QNNOptions{Ranges: qnnRanges()})
	if report.Activations != quantize.QNNUint8 || len(report.Quantized) != 5 || len(report.Skipped) != 0 {
		t.Fatalf("report = %+v", report)
	}

	// c1 in [-6, 6]: scale 12/255, zero point round(6 / scale) = 128.
	c1, ok := tensorParams(report, "c1")
	if !ok || c1.Source != "range" || c1.Type != quantize.QNNUint8 || math.Abs(float64(c1.Scale)-12.0/255) > 1e-7 || c1.ZeroPoint != 128 {
		t.Errorf("c1 = %+v", c1)
	}
	// m1 in [-0.3, 6]: the zero point is round(0.3 / (6.3 / 255)) = 12.
	if m1, _ := tensorParams(report, "m1"); m1.ZeroPoint != 12 || math.Abs(float64(m1.Scale)-6.3/255) > 1e-7 {
		t.Errorf("m1 = %+v", m1)
	}
	// Sigmoid has the scale the HTP requires, whatever range was measured.
	if s1, _ := tensorParams(report, "s1"); s1.Source != "operator" || s1.Scale != 1.0/256 || s1.ZeroPoint != 0 {
		t.Errorf("s1 = %+v", s1)
	}
	// MaxPool moves data: its output is quantized as its input.
	m1, _ := tensorParams(report, "m1")
	if p1, _ := tensorParams(report, "p1"); p1.Source != "input" || p1.Scale != m1.Scale || p1.ZeroPoint != m1.ZeroPoint {
		t.Errorf("p1 = %+v, m1 = %+v", p1, m1)
	}

	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	g := m.Graph
	counts := g.OpCounts()
	if counts["QuantizeLinear"] != 6 || counts["DequantizeLinear"] != 6+2+2 || counts["Conv"] != 2 {
		t.Errorf("ops %v", counts)
	}
	// The weight is uint8, per output channel, zero point 128, scale 2 max|w| / 255; the bias is int32 with the product scale.
	w1 := g.Initializer("w1_quantized")
	if w1 == nil || w1.DataType != graph.Uint8 || len(w1.Dims) != 4 {
		t.Fatalf("w1_quantized = %+v", w1)
	}
	scales, err := g.Initializer("w1_scale").Float32s()
	if err != nil || len(scales) != 4 {
		t.Fatalf("w1_scale = %v, %v", scales, err)
	}
	values := pseudoRandom(21, 4*3*3*3, 0.4)
	for c := range 4 {
		peak := 0.0
		for _, v := range values[c*27 : (c+1)*27] {
			peak = math.Max(peak, math.Abs(float64(v)))
		}
		if math.Abs(float64(scales[c])-2*peak/255) > 1e-7 {
			t.Errorf("channel %d scale %v, want %v", c, scales[c], 2*peak/255)
		}
	}
	b1 := g.Initializer("b1_quantized")
	bs, _ := g.Initializer("b1_quantized_scale").Float32s()
	if b1 == nil || b1.DataType != graph.Int32 || len(bs) != 4 {
		t.Fatalf("b1_quantized = %+v", b1)
	}
	x, _ := tensorParams(report, "x")
	for c := range 4 {
		if want := x.Scale * scales[c]; math.Abs(float64(bs[c]-want)) > 1e-9 {
			t.Errorf("bias scale %v, want %v", bs[c], want)
		}
		got := int32(binary.LittleEndian.Uint32(b1.RawData[4*c:]))
		bias := pseudoRandom(22, 4, 0.1)[c]
		if want := math.RoundToEven(float64(bias / bs[c])); math.Abs(float64(got)-want) > 0 {
			t.Errorf("bias %d = %d, want %v", c, got, want)
		}
	}
	// The float weights and biases nothing reads are gone, and the graph keeps its names.
	if g.Initializer("w1") != nil || g.Initializer("b1") != nil || len(report.RemovedInitializers) != 4 {
		t.Errorf("removed %v", report.RemovedInitializers)
	}
	if !g.IsOutput("y") || len(g.Outputs) != 1 {
		t.Errorf("outputs %v", g.Outputs)
	}

	// It is not the form TensorRT runs.
	if err := quantize.CheckTensorRT(m); !errors.Is(err, quantize.ErrNotTensorRTQDQ) {
		t.Errorf("CheckTensorRT = %v, want ErrNotTensorRTQDQ", err)
	}
}

func TestQNNRunsCloseToTheFloatModelAndSixteenBitsRunsCloser(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	float := qnnNet()

	run := func(opts quantize.QNNOptions) float64 {
		body, _ := qnnExport(t, opts)
		worst, peak := 0.0, 0.0
		for seed := range uint64(6) {
			input := randomTensor(300+seed, 1, 3, 8, 8)
			want := runOnce(t, rt, qnnManifest(onnx.Float32), float, input)
			got := runOnce(t, rt, qnnManifest(onnx.Float32), body, input)
			worst = math.Max(worst, maxAbsDiff(want, got))
			for _, v := range want.Float32 {
				peak = math.Max(peak, math.Abs(float64(v)))
			}
		}
		return worst / peak
	}
	eight := run(quantize.QNNOptions{Ranges: qnnRanges()})
	sixteen := run(quantize.QNNOptions{Ranges: qnnRanges(), Activations: quantize.QNNUint16})
	if eight > 0.08 {
		t.Errorf("uint8 differs from the float model by %.3f of the peak", eight)
	}
	if sixteen > 0.02 || sixteen > eight {
		t.Errorf("uint16 differs by %.4f of the peak, uint8 by %.4f", sixteen, eight)
	}

	// uint16 is written in com.microsoft, which the model imports.
	m, err := graph.Load(func() []byte {
		b, _ := qnnExport(t, quantize.QNNOptions{Ranges: qnnRanges(), Activations: quantize.QNNUint16})
		return b
	}())
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := m.Opset("com.microsoft"); !ok || v != 1 {
		t.Errorf("com.microsoft opset = %d, %v", v, ok)
	}
	for _, n := range m.Graph.Nodes {
		if (n.OpType == "QuantizeLinear" || n.OpType == "DequantizeLinear") && n.Domain != "com.microsoft" && !strings.HasPrefix(n.Name, "w") && !strings.HasPrefix(n.Name, "b") {
			t.Fatalf("%s %q is in domain %q", n.OpType, n.Name, n.Domain)
		}
	}
}

func TestQNNReadsAnIntegerInputWithNoQuantizeLinear(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	ranges := qnnRanges()
	ranges["x"] = quantize.Range{Min: 0, Max: 1}
	body, report := qnnExport(t, quantize.QNNOptions{
		Ranges:        ranges,
		Fixed:         map[string]quantize.QParams{"x": {Type: quantize.QNNUint8, Scale: 1.0 / 255}},
		IntegerInputs: []string{"x"},
	})
	if x, _ := tensorParams(report, "x"); x.Source != "fixed" || x.Scale != 1.0/255 || x.ZeroPoint != 0 {
		t.Errorf("x = %+v", x)
	}
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if typ, ok := m.Graph.TypeOf("x"); !ok || typ.ElemType != graph.Uint8 {
		t.Fatalf("input type %+v", typ)
	}
	for _, n := range m.Graph.Nodes {
		if n.OpType == "QuantizeLinear" && n.Inputs[0] == "x" {
			t.Fatal("the integer input still has a QuantizeLinear")
		}
	}

	// Bytes in, and the float model on byte/255: the same function.
	pixels := make([]uint8, 3*8*8)
	floats := make([]float32, len(pixels))
	for i := range pixels {
		pixels[i] = uint8((i*37 + 11) % 256)
		floats[i] = float32(pixels[i]) / 255
	}
	want := runOnce(t, rt, qnnManifest(onnx.Float32), qnnNet(), onnx.Tensor{Shape: []int64{1, 3, 8, 8}, Float32: floats})
	got := runOnce(t, rt, qnnManifest(onnx.Uint8), body, onnx.Tensor{Shape: []int64{1, 3, 8, 8}, Uint8: pixels})
	peak := 0.0
	for _, v := range want.Float32 {
		peak = math.Max(peak, math.Abs(float64(v)))
	}
	if d := maxAbsDiff(want, got); d > 0.08*peak {
		t.Errorf("the integer-input model differs from the float model by %g of %g", d, peak)
	}
}

func TestQNNWeightScalesMakeAMeanWeightExact(t *testing.T) {
	g := message{}.
		sub(1, node("Conv", []string{"x", "mean"}, []string{"y"}, intsAttribute("kernel_shape", 2, 2), intsAttribute("group", 2))).
		text(2, "mean").
		sub(5, floatInitializer("mean", []float32{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25}, 2, 1, 2, 2)).
		sub(11, valueInfo("x", protoFloat, 1, 2, 4, 4)).
		sub(12, valueInfo("y", protoFloat, 1, 2, 3, 3))
	body := modelAt(g, 8, 17)
	ranges := map[string]quantize.Range{"x": {Min: 0, Max: 1}, "y": {Min: 0, Max: 1}}
	for name, tc := range map[string]struct {
		scales map[string]float32
		exact  bool
	}{"default scale": {nil, false}, "power of two": {map[string]float32{"mean": 1.0 / 256}, true}} {
		t.Run(name, func(t *testing.T) {
			m, err := graph.Load(body)
			if err != nil {
				t.Fatal(err)
			}
			report, err := quantize.QNN(m, quantize.QNNOptions{Ranges: ranges, WeightScales: tc.scales})
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Weights) != 1 {
				t.Fatalf("weights %+v", report.Weights)
			}
			if exact := report.Weights[0].MaxError == 0; exact != tc.exact {
				t.Errorf("max error %g, exact = %v, want %v", report.Weights[0].MaxError, exact, tc.exact)
			}
		})
	}
}

func TestQNNLeavesWhatItCannotQuantizeInFloatAndRefusesWhatWouldBeWrong(t *testing.T) {
	t.Run("a node without a range stays in float", func(t *testing.T) {
		ranges := qnnRanges()
		delete(ranges, "y")
		_, report := qnnExport(t, quantize.QNNOptions{Ranges: ranges})
		if len(report.Skipped) != 1 || report.Skipped[0].OpType != "Conv" || !strings.Contains(report.Skipped[0].Reason, `"y"`) {
			t.Errorf("skipped %+v", report.Skipped)
		}
	})
	t.Run("excluded nodes stay in float", func(t *testing.T) {
		m, _ := graph.Load(qnnNet())
		ranges := qnnRanges()
		ranges["p1"] = quantize.Range{Min: -0.3, Max: 6}
		report, err := quantize.QNN(m, quantize.QNNOptions{Ranges: ranges, ExcludeOpTypes: []string{"MaxPool"}})
		if err != nil {
			t.Fatal(err)
		}
		if len(report.Skipped) != 1 || report.Skipped[0].OpType != "MaxPool" {
			t.Errorf("skipped %+v", report.Skipped)
		}
	})
	for name, tc := range map[string]struct {
		opts quantize.QNNOptions
		want string
	}{
		"no ranges at all":               {quantize.QNNOptions{}, "no node is left"},
		"a bad activation":               {quantize.QNNOptions{Ranges: qnnRanges(), Activations: "int8"}, "activations"},
		"an inverted range":              {quantize.QNNOptions{Ranges: map[string]quantize.Range{"x": {Min: 1, Max: -1}}}, "range"},
		"a zero scale":                   {quantize.QNNOptions{Ranges: qnnRanges(), Fixed: map[string]quantize.QParams{"c1": {Scale: 0}}}, "scale"},
		"a zero point too big":           {quantize.QNNOptions{Ranges: qnnRanges(), Fixed: map[string]quantize.QParams{"c1": {Scale: 1, ZeroPoint: 256}}}, "zero point"},
		"an integer input without a pin": {quantize.QNNOptions{Ranges: qnnRanges(), IntegerInputs: []string{"x"}}, "Fixed"},
		"an integer input that is not one": {quantize.QNNOptions{Ranges: qnnRanges(), IntegerInputs: []string{"c1"},
			Fixed: map[string]quantize.QParams{"c1": {Scale: 1}}}, "graph input"},
	} {
		t.Run(name, func(t *testing.T) {
			body := qnnNet()
			m, err := graph.Load(body)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := quantize.QNN(m, tc.opts); !errors.Is(err, quantize.ErrRefused) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("QNN = %v, want ErrRefused naming %q", err, tc.want)
			}
			out, _ := m.Marshal()
			if !bytes.Equal(out, body) {
				t.Error("a refused model was changed")
			}
		})
	}

	t.Run("a bias that overflows int32 names the range to widen", func(t *testing.T) {
		ranges := qnnRanges()
		// Conv 1 has a bias around 0.1; an input scale this small overflows it.
		ranges["x"] = quantize.Range{Min: 0, Max: 1e-9}
		m, _ := graph.Load(qnnNet())
		_, err := quantize.QNN(m, quantize.QNNOptions{Ranges: ranges, Fixed: map[string]quantize.QParams{"x": {Scale: 1e-12}}})
		if !errors.Is(err, quantize.ErrRefused) || !strings.Contains(err.Error(), "overflows int32") || !strings.Contains(err.Error(), `widen the range of "x"`) {
			t.Fatalf("QNN = %v", err)
		}
	})
	t.Run("a model already quantized is refused", func(t *testing.T) {
		body, _ := qnnExport(t, quantize.QNNOptions{Ranges: qnnRanges()})
		m, _ := graph.Load(body)
		if _, err := quantize.QNN(m, quantize.QNNOptions{Ranges: qnnRanges()}); !errors.Is(err, quantize.ErrRefused) || !strings.Contains(err.Error(), "already quantized") {
			t.Fatalf("QNN = %v", err)
		}
	})
	t.Run("the digest of the scales is stable", func(t *testing.T) {
		_, a := qnnExport(t, quantize.QNNOptions{Ranges: qnnRanges()})
		_, b := qnnExport(t, quantize.QNNOptions{Ranges: qnnRanges()})
		_, c := qnnExport(t, quantize.QNNOptions{Ranges: qnnRanges(), Activations: quantize.QNNUint16})
		if a.ScalesSHA256 != b.ScalesSHA256 || a.ScalesSHA256 == c.ScalesSHA256 || len(a.ScalesSHA256) != 64 {
			t.Errorf("digests %s %s %s", a.ScalesSHA256, b.ScalesSHA256, c.ScalesSHA256)
		}
	})
}

func TestTheQuantizeCommandWritesTheQNNFormOnlyWhenAskedAndTheOtherModeIsUnchanged(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	binary := buildCommand(t, "arandu-onnx-quantize")

	source := qnnManifest(onnx.Float32)
	manifest, dir := writeModel(t, source, qnnNet())
	manifestPath := writeManifest(t, manifest, dir)

	options := filepath.Join(dir, "qnn.json")
	ranges := qnnRanges()
	ranges["x"] = quantize.Range{Min: 0, Max: 1}
	writeJSONFile(t, options, quantize.QNNOptions{Ranges: ranges})
	out := filepath.Join(dir, "qnnnet.qnn.onnx")
	stdout, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-library", rt.Config().SharedLibraryPath,
		"-qnn", options, "-low", "0", "-high", "1", "-out", out)
	if err != nil {
		t.Fatalf("running the command: %v\n%s", err, stderr)
	}
	printed, err := onnx.LoadManifest(stdout)
	if err != nil {
		t.Fatalf("the printed manifest %s: %v", stdout, err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if printed.Precision != onnx.PrecisionINT8 || printed.Quantization != "" || printed.SHA256 != sha(body) ||
		printed.DerivedFrom == nil || printed.DerivedFrom.ID != "qnnnet" || !strings.Contains(printed.DerivedFrom.Transform, "QNN") {
		t.Errorf("printed = %+v, derived from %+v", printed, printed.DerivedFrom)
	}
	if !strings.Contains(stderr, "compared on the CPU") {
		t.Errorf("stderr = %s", stderr)
	}
	// The printed manifest opens the file.
	if _, err := rt.Open(printed, dir); err != nil {
		t.Errorf("opening the exported model through its manifest: %v", err)
	}

	// An integer input is declared uint8 in the manifest, and Open checks it.
	writeJSONFile(t, options, quantize.QNNOptions{
		Ranges:        ranges,
		Fixed:         map[string]quantize.QParams{"x": {Type: quantize.QNNUint8, Scale: 1.0 / 255}},
		IntegerInputs: []string{"x"},
	})
	byteOut := filepath.Join(dir, "qnnnet.bytes.onnx")
	stdout, stderr, err = runCommand(t, binary, "-manifest", manifestPath, "-library", rt.Config().SharedLibraryPath,
		"-qnn", options, "-out", byteOut)
	if err != nil {
		t.Fatalf("running the command: %v\n%s", err, stderr)
	}
	printed, err = onnx.LoadManifest(stdout)
	if err != nil || printed.Inputs[0].ElementType != onnx.Uint8 {
		t.Fatalf("printed = %+v, %v", printed, err)
	}
	if _, err := rt.Open(printed, dir); err != nil {
		t.Errorf("opening the integer-input model through its manifest: %v", err)
	}

	// Unknown keys in the options are refused, and nothing is written.
	if err := os.WriteFile(options, []byte(`{"ranges": {}, "rangez": {}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	never := filepath.Join(dir, "never.onnx")
	if _, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-qnn", options, "-out", never); err == nil || !strings.Contains(stderr, "rangez") {
		t.Errorf("unknown key: %v, %s", err, stderr)
	}
	if _, err := os.Stat(never); !os.IsNotExist(err) {
		t.Errorf("a refused export was written: %v", err)
	}
}

func writeJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	body, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
}
