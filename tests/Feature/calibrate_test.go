package feature_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
	"github.com/hyz-is/arandu-onnx/internal/onnxgraph"
)

// The affine model is y = x * w + b, label = argmax(y), with
// w = {{1, 2}, {3, 4}, {5, 6}} and b = {0.5, -0.5}. Every range below is
// worked out by hand from that, not by running the model.

// affineSamples are two batches of the affine model's input.
//
// Batch one, x = {{1, 0, 0}, {0, 1, 1}}: xw = {{1, 2}, {8, 10}}, y = {{1.5,
// 1.5}, {8.5, 9.5}}. Batch two, x = {{-2, 0.5, 0}}: xw = {{-0.5, -2}}, y =
// {{0, -2.5}}. So |x| reaches 2, |xw| 10 and |y| 9.5.
func affineSamples() []map[string]onnx.Tensor {
	return []map[string]onnx.Tensor{
		{"x": {Shape: []int64{2, 3}, Float32: []float32{1, 0, 0, 0, 1, 1}}},
		{"x": {Shape: []int64{1, 3}, Float32: []float32{-2, 0.5, 0}}},
	}
}

func TestTheGraphReaderSeesTheActivations(t *testing.T) {
	t.Parallel()

	g, err := onnxgraph.Read(affineModel())
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got := strings.Join(g.Activations(), ","); got != "x,xw,y,label" {
		t.Errorf("Activations = %s, want x,xw,y,label (w and b are initializers)", got)
	}
	if len(g.Nodes) != 3 || g.Nodes[0].OpType != "MatMul" || g.Nodes[2].Outputs[0] != "label" {
		t.Errorf("Nodes = %+v", g.Nodes)
	}

	extended, err := onnxgraph.WithOutputs(affineModel(), []string{"xw", "y", "xw"})
	if err != nil {
		t.Fatalf("WithOutputs: %v", err)
	}
	g2, err := onnxgraph.Read(extended)
	if err != nil {
		t.Fatalf("Read(extended): %v", err)
	}
	if got := strings.Join(g2.Outputs, ","); got != "y,label,xw" {
		t.Errorf("Outputs = %s, want y,label,xw: an output is added once", got)
	}

	for name, body := range map[string][]byte{
		"empty":     {},
		"truncated": affineModel()[:20],
		"no graph":  message{}.integer(1, 7),
	} {
		if _, err := onnxgraph.Read(body); !errors.Is(err, onnxgraph.ErrMalformed) {
			t.Errorf("%s: Read = %v, want ErrMalformed", name, err)
		}
	}
}

func TestCalibrationMeasuresEveryActivation(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())

	result, err := runtime.Calibrate(context.Background(), manifest, dir, onnx.Samples(affineSamples()...), calibration.Options{})
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	want := calibration.Table{"x": 2, "xw": 10, "y": 9.5}
	if len(result.Table) != len(want) {
		t.Fatalf("Table = %v, want %v", result.Table, want)
	}
	for name, r := range want {
		if result.Table[name] != r {
			t.Errorf("range of %s = %v, want %v", name, result.Table[name], r)
		}
	}
	if result.Samples != 2 || strings.Join(result.NotFloat, ",") != "label" {
		t.Errorf("Samples = %d, NotFloat = %v; want 2 and the int64 label", result.Samples, result.NotFloat)
	}

	excluded, err := runtime.Calibrate(context.Background(), manifest, dir, onnx.Samples(affineSamples()...), calibration.Options{
		Exclude: []string{"y"},
	})
	if err != nil {
		t.Fatalf("Calibrate with an exclusion: %v", err)
	}
	if _, present := excluded.Table["y"]; present || strings.Join(excluded.Excluded, ",") != "y" {
		t.Errorf("Table = %v, Excluded = %v; want y left out", excluded.Table, excluded.Excluded)
	}
}

func TestTheHistogramMethodsReadTheSamplesTwice(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())

	var samples []map[string]onnx.Tensor
	for i := range 64 {
		v := float32(i%16) / 4
		samples = append(samples, map[string]onnx.Tensor{"x": {Shape: []int64{1, 3}, Float32: []float32{v, -v / 2, 1}}})
	}
	for _, method := range []calibration.Method{calibration.MethodPercentile, calibration.MethodEntropy} {
		result, err := runtime.Calibrate(context.Background(), manifest, dir, onnx.Samples(samples...), calibration.Options{Method: method})
		if err != nil {
			t.Fatalf("%s: Calibrate: %v", method, err)
		}
		// |x| reaches 3.75; every threshold lies in (0, absmax].
		if r := result.Table["x"]; r <= 0 || r > 3.75 {
			t.Errorf("%s: range of x = %v, want in (0, 3.75]", method, r)
		}
		if result.Report.Method != method {
			t.Errorf("Report.Method = %s, want %s", result.Report.Method, method)
		}
	}

	// A sequence that yields something else the second time is refused.
	calls := 0
	shifting := func(yield func(map[string]onnx.Tensor, error) bool) {
		calls++
		for _, s := range samples[:calls] {
			if !yield(s, nil) {
				return
			}
		}
	}
	if _, err := runtime.Calibrate(context.Background(), manifest, dir, shifting, calibration.Options{Method: calibration.MethodEntropy}); err == nil {
		t.Error("Calibrate accepted samples that changed between passes")
	}
}

func TestCalibrationHoldsTheModelToItsManifest(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	ctx := context.Background()
	samples := onnx.Samples(affineSamples()...)

	blocked := manifest
	blocked.Status = onnx.StatusBlocked
	if _, err := runtime.Calibrate(ctx, blocked, dir, samples, calibration.Options{}); !errors.Is(err, onnx.ErrModelBlocked) {
		t.Errorf("blocked: Calibrate = %v, want ErrModelBlocked", err)
	}
	tampered := manifest
	tampered.SHA256 = strings.Repeat("0", 64)
	if _, err := runtime.Calibrate(ctx, tampered, dir, samples, calibration.Options{}); !errors.Is(err, onnx.ErrDigestMismatch) {
		t.Errorf("tampered: Calibrate = %v, want ErrDigestMismatch", err)
	}
	half := manifest
	half.Precision = onnx.PrecisionFP16
	if _, err := runtime.Calibrate(ctx, half, dir, samples, calibration.Options{}); err == nil {
		t.Error("Calibrate accepted a model that is not float32")
	}
	if _, err := runtime.Calibrate(ctx, manifest, dir, onnx.Samples(), calibration.Options{}); err == nil {
		t.Error("Calibrate accepted no samples")
	}
	wrong := onnx.Samples(map[string]onnx.Tensor{"x": {Shape: []int64{1, 4}, Float32: []float32{1, 2, 3, 4}}})
	if _, err := runtime.Calibrate(ctx, manifest, dir, wrong, calibration.Options{}); !errors.Is(err, onnx.ErrInvalidTensor) {
		t.Errorf("a sample of the wrong shape: Calibrate = %v, want ErrInvalidTensor", err)
	}
}

// withTable calibrates the affine model, writes the table beside it, and
// returns the manifest pinning it.
func withTable(t *testing.T, runtime *onnx.Runtime, format calibration.Format) (onnx.Manifest, string, []byte) {
	t.Helper()
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	result, err := runtime.Calibrate(context.Background(), manifest, dir, onnx.Samples(affineSamples()...), calibration.Options{})
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	table, err := result.Table.Marshal(format)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "affine.table"), table, 0o600); err != nil {
		t.Fatalf("writing the table: %v", err)
	}
	manifest.Calibration = &onnx.Calibration{
		File:      "affine.table",
		SHA256:    sha(table),
		Format:    format,
		Precision: onnx.PrecisionINT8,
		Method:    onnx.CalibrationMinMax,
		Source:    "the affine test model over two hand-written batches",
	}
	return manifest, dir, table
}

func TestAModelOpensWithItsVerifiedTable(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	for _, format := range []calibration.Format{calibration.FormatONNXRuntime, calibration.FormatTensorRT} {
		manifest, dir, _ := withTable(t, runtime, format)
		session, err := runtime.Open(manifest, dir)
		if err != nil {
			t.Fatalf("%s: Open: %v", format, err)
		}
		// On the CPU nobody reads the table, and the evidence says so.
		if info := session.Model(); info.Provider != onnx.ProviderCPU || info.Calibration != "" {
			t.Errorf("%s: ModelInfo = %+v, want cpu and no calibration", format, info)
		}
		if got := session.Manifest().Calibration; got == nil || got.SHA256 != manifest.Calibration.SHA256 {
			t.Errorf("%s: Manifest().Calibration = %+v", format, got)
		}
	}
}

func TestATableThatIsNotTheManifestsIsRefused(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir, _ := withTable(t, runtime, calibration.FormatONNXRuntime)

	tampered := manifest
	c := *manifest.Calibration
	c.SHA256 = strings.Repeat("a", 64)
	tampered.Calibration = &c
	if _, err := runtime.Open(tampered, dir); !errors.Is(err, onnx.ErrDigestMismatch) {
		t.Errorf("a table with another digest: Open = %v, want ErrDigestMismatch", err)
	}

	// A table whose bytes match its digest and do not parse.
	garbage := []byte("not a table")
	if err := os.WriteFile(filepath.Join(dir, "garbage.table"), garbage, 0o600); err != nil {
		t.Fatal(err)
	}
	broken := manifest
	b := *manifest.Calibration
	b.File, b.SHA256 = "garbage.table", sha(garbage)
	broken.Calibration = &b
	if _, err := runtime.Open(broken, dir); !errors.Is(err, calibration.ErrInvalidTable) {
		t.Errorf("an unparseable table: Open = %v, want ErrInvalidTable", err)
	}

	// A well-formed table of another model.
	foreign, err := calibration.Table{"/backbone/conv1/Conv": 4}.MarshalONNXRuntime()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "foreign.table"), foreign, 0o600); err != nil {
		t.Fatal(err)
	}
	other := manifest
	f := *manifest.Calibration
	f.File, f.SHA256 = "foreign.table", sha(foreign)
	other.Calibration = &f
	if _, err := runtime.Open(other, dir); !errors.Is(err, onnx.ErrContractMismatch) {
		t.Errorf("a table of another model: Open = %v, want ErrContractMismatch", err)
	}

	missing := manifest
	m := *manifest.Calibration
	m.File = "absent.table"
	missing.Calibration = &m
	if _, err := runtime.Open(missing, dir); err == nil {
		t.Error("Open accepted a manifest whose table is missing")
	}
}

// sha is the lowercase hexadecimal SHA-256 of the bytes.
func sha(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// scalarConstantModel is y = x^2 + 2*2, with the exponent and the term
// computed from scalar Constant nodes, as some exporters write them.
func scalarConstantModel() []byte {
	scalar := func(name string, v float32) message {
		return node("Constant", nil, []string{name}, message{}.text(1, "value").sub(5, floatInitializer(name+"_value", []float32{v})).integer(20, 4))
	}
	g := message{}.
		sub(1, scalar("e", 2)).
		sub(1, node("Pow", []string{"x", "e"}, []string{"p"})).
		sub(1, node("Mul", []string{"e", "e"}, []string{"e2"})).
		sub(1, node("Add", []string{"p", "e2"}, []string{"y"})).
		text(2, "scalar-constants").
		sub(11, valueInfo("x", protoFloat, -1, 2)).
		sub(12, valueInfo("y", protoFloat, -1, 2))
	return model(g)
}

func TestCalibrationLeavesOutWhatConstantsCompute(t *testing.T) {
	g, err := onnxgraph.Read(scalarConstantModel())
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got := strings.Join(g.Activations(), ","); got != "x,p,y" {
		t.Errorf("Activations = %s, want x,p,y: e is a Constant node and e2 is computed from constants only", got)
	}

	// The scalars are values ONNX Runtime's Go binding cannot return as
	// outputs; calibrating them used to end the process.
	rt := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, syntheticManifest("scalars", "x", 2, "y", 2), scalarConstantModel())
	result, err := rt.Calibrate(context.Background(), manifest, dir, onnx.Samples(
		map[string]onnx.Tensor{"x": {Shape: []int64{1, 2}, Float32: []float32{1, -3}}},
	), calibration.Options{})
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	// |x| reaches 3, p = x^2 reaches 9 and y = p + 4 reaches 13.
	if len(result.Table) != 3 || result.Table["x"] != 3 || result.Table["p"] != 9 || result.Table["y"] != 13 {
		t.Errorf("table = %v, want x 3, p 9, y 13", result.Table)
	}
}
