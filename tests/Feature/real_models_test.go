package feature_test

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/rewrite"
)

// The real-model checks rewrite exported models nobody committed here and
// compare them with the originals on ONNX Runtime. Where the models and
// their samples are is the machine's business, so they are named only by
// these variables, read by the tests and by nothing else, and each check
// skips when its model is not named:
//
//	ARANDU_ONNX_OPSET_UPGRADE_MODEL      an export below opset 13 with one float NCHW input
//	ARANDU_ONNX_OPSET_UPGRADE_SAMPLES    a directory of JPEG images to feed it (optional)
//	ARANDU_ONNX_WEIGHT_INPUTS_MODEL      an export that lists its weights as graph inputs
//	ARANDU_ONNX_WEIGHT_INPUTS_SAMPLES    a directory of JPEG images to feed it (optional)
//
// Synthetic inputs are always fed as well. Nothing a check reads or writes
// leaves the test's temporary directory.

// realModel returns the model a variable names, or skips the test.
func realModel(t *testing.T, variable string) []byte {
	t.Helper()
	path := os.Getenv(variable)
	if path == "" {
		t.Skipf("%s is not set: this check runs on a real model the machine provides", variable)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%s: %v", variable, err)
	}
	return body
}

// realSamples returns up to n JPEG files of the directory a variable names,
// in name order, or none when it is not set.
func realSamples(t *testing.T, variable string, n int) []string {
	t.Helper()
	dir := os.Getenv(variable)
	if dir == "" {
		return nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("%s: %v", variable, err)
	}
	var out []string
	for _, e := range entries {
		if name := strings.ToLower(e.Name()); strings.HasSuffix(name, ".jpg") || strings.HasSuffix(name, ".jpeg") {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	slices.Sort(out)
	return out[:min(n, len(out))]
}

// contract is the manifest of a real model, read from its graph: its one
// float input and its float outputs, with their ranks and fixed dimensions.
func contract(t *testing.T, g *graph.Graph, id string) (onnx.Manifest, []graph.Dim) {
	t.Helper()
	var input *graph.ValueInfo
	for _, v := range g.Inputs {
		if g.Initializer(v.Name) != nil {
			continue
		}
		if input != nil {
			t.Skipf("the model has more than one input (%s, %s); the check feeds one", input.Name, v.Name)
		}
		input = v
	}
	if input == nil {
		t.Fatal("the model has no input")
	}
	in, ok := input.Tensor()
	if !ok || !in.Ranked || in.ElemType != graph.Float || len(in.Shape) != 4 {
		t.Skipf("the input %s is not a float NCHW tensor the graph records", input.Name)
	}
	dims := func(shape []graph.Dim) []int64 {
		out := make([]int64, len(shape))
		for i, d := range shape {
			out[i] = -1
			if d.Value > 0 {
				out[i] = d.Value
			}
		}
		return out
	}
	m := syntheticManifest(id, input.Name, 4, "unused", 1)
	m.Inputs[0].Shape = dims(in.Shape)
	m.Outputs = nil
	for _, v := range g.Outputs {
		out, ok := v.Tensor()
		if !ok || out.ElemType != graph.Float || !out.Ranked {
			continue
		}
		m.Outputs = append(m.Outputs, onnx.TensorSpec{Name: v.Name, ElementType: onnx.Float32, Shape: dims(out.Shape)})
	}
	if len(m.Outputs) == 0 {
		t.Skip("the model has no float output the graph records")
	}
	return m, in.Shape
}

// realInputs is three synthetic inputs and one per sample, shaped as the
// input's fixed dimensions say: batch 1, a dynamic height or width taken
// from the image (keeping its aspect when the other is fixed), 64 for the
// synthetic ones.
func realInputs(t *testing.T, shape []graph.Dim, samples []string) []onnx.Tensor {
	t.Helper()
	fixed := func(i int, otherwise int64) int64 {
		if shape[i].Value > 0 {
			return shape[i].Value
		}
		return otherwise
	}
	if fixed(1, 3) != 3 {
		t.Skipf("the input has %d channels; the samples are RGB", shape[1].Value)
	}
	var out []onnx.Tensor
	for seed := range uint64(3) {
		out = append(out, randomTensor(900+seed, 1, 3, fixed(2, 64), fixed(3, 64)))
	}
	for _, path := range samples {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		img, err := onnx.DecodeImage(f)
		_ = f.Close()
		if err != nil {
			t.Fatal(err)
		}
		w, h := int64(img.Rect.Dx()), int64(img.Rect.Dy())
		height, width := fixed(2, 0), fixed(3, 0)
		switch {
		case height == 0 && width == 0:
			height, width = h, w
		case height == 0:
			height = max(8, int64(math.Ceil(float64(width)*float64(h)/float64(w))))
		case width == 0:
			width = max(8, int64(math.Ceil(float64(height)*float64(w)/float64(h))))
		}
		resized, err := onnx.Resize(img, int(width), int(height))
		if err != nil {
			t.Fatal(err)
		}
		x, err := onnx.NCHW(resized, onnx.Normalization{Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}})
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, x)
	}
	return out
}

// compareReal runs the original and the rewritten model on the inputs and
// returns, over every float output, the largest difference, the largest
// |original|, and the lowest cosine between an original and a rewritten
// output.
func compareReal(t *testing.T, m onnx.Manifest, original, rewritten []byte, inputs []onnx.Tensor) (float64, float64, float64) {
	t.Helper()
	rt := newRuntime(t, onnx.Config{})
	am, adir := writeModel(t, m, original)
	bm := m
	bm.ID += ".rewritten"
	bm, bdir := writeModel(t, bm, rewritten)
	a, err := rt.Open(am, adir)
	if err != nil {
		t.Fatalf("opening the original: %v", err)
	}
	defer func() { _ = a.Close() }()
	b, err := rt.Open(bm, bdir)
	if err != nil {
		t.Fatalf("opening the rewritten model: %v", err)
	}
	defer func() { _ = b.Close() }()

	worst, peak, lowest := 0.0, 0.0, 1.0
	for _, x := range inputs {
		feed := map[string]onnx.Tensor{m.Inputs[0].Name: x}
		want, err := a.Run(t.Context(), feed)
		if err != nil {
			t.Fatal(err)
		}
		got, err := b.Run(t.Context(), feed)
		if err != nil {
			t.Fatal(err)
		}
		for _, spec := range m.Outputs {
			w, g := want.Outputs[spec.Name], got.Outputs[spec.Name]
			worst = math.Max(worst, maxAbsDiff(w, g))
			for _, v := range w.Float32 {
				peak = math.Max(peak, math.Abs(float64(v)))
			}
			lowest = math.Min(lowest, cosine(w.Float32, g.Float32))
		}
	}
	return worst, peak, lowest
}

// rewriteReal applies the rewrites to a copy of the model.
func rewriteReal(t *testing.T, body []byte, apply func(*graph.Model) error) (*graph.Model, []byte) {
	t.Helper()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := apply(m); err != nil {
		t.Fatal(err)
	}
	out, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(out, body) {
		t.Fatal("the rewrite left the model as it was")
	}
	return m, out
}

func TestARealExportUpgradedToOpset13ComputesTheSame(t *testing.T) {
	body := realModel(t, "ARANDU_ONNX_OPSET_UPGRADE_MODEL")
	var report rewrite.OpsetReport
	m, out := rewriteReal(t, body, func(m *graph.Model) error {
		var err error
		report, err = rewrite.UpgradeOpset(m, rewrite.OpsetOptions{})
		return err
	})
	if report.OpsetAfter != 13 || report.OpsetBefore >= 13 {
		t.Fatalf("report = %+v", report)
	}
	changes := map[string]int{}
	for _, c := range report.Converted {
		changes[c.OpType]++
	}
	t.Logf("opset %d to %d, IR %d to %d, %d nodes before and %d after; converted %v",
		report.OpsetBefore, report.OpsetAfter, report.IRBefore, report.IRAfter, report.NodesBefore, report.NodesAfter, changes)

	manifest, shape := contract(t, m.Graph, "real.opset-upgrade")
	inputs := realInputs(t, shape, realSamples(t, "ARANDU_ONNX_OPSET_UPGRADE_SAMPLES", 8))
	worst, peak, _ := compareReal(t, manifest, body, out, inputs)
	t.Logf("over %d inputs: max |original - upgraded| = %g, max |original| = %g", len(inputs), worst, peak)
	if worst > 1e-5*math.Max(1, peak) {
		t.Errorf("the upgraded model differs from the original by %g, more than 1e-5 of %g", worst, math.Max(1, peak))
	}
}

func TestARealExportWithConstantWeightsAndNoDropoutComputesTheSame(t *testing.T) {
	body := realModel(t, "ARANDU_ONNX_WEIGHT_INPUTS_MODEL")
	var inits rewrite.InitializersReport
	var dropout rewrite.DropoutReport
	m, out := rewriteReal(t, body, func(m *graph.Model) error {
		var err error
		if inits, err = rewrite.InitializersOutOfInputs(m); err != nil {
			return err
		}
		dropout, err = rewrite.DropoutToIdentity(m)
		return err
	})
	if len(inits.Removed) == 0 {
		t.Fatal("the model lists no initializer as a graph input")
	}
	t.Logf("%d initializers out of the inputs (%d to %d inputs), %d Dropout to Identity",
		len(inits.Removed), inits.InputsBefore, inits.InputsAfter, len(dropout.Replaced))

	manifest, shape := contract(t, m.Graph, "real.weight-inputs")
	inputs := realInputs(t, shape, realSamples(t, "ARANDU_ONNX_WEIGHT_INPUTS_SAMPLES", 8))
	worst, _, lowest := compareReal(t, manifest, body, out, inputs)
	t.Logf("over %d inputs: 1 - lowest cosine = %.3g, max |original - rewritten| = %g", len(inputs), 1-lowest, worst)
	if lowest < 1-1e-6 {
		t.Errorf("an output of the rewritten model has cosine %.9f with the original's, below 1 - 1e-6", lowest)
	}
}

// cosine is the cosine of the angle between two vectors.
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 && nb == 0 {
		return 1
	}
	return dot / math.Sqrt(na*nb)
}
