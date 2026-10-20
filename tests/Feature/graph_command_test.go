package feature_test

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/quantize"
)

// buildCommand builds one of the commands in cmd/ into a temporary
// directory.
func buildCommand(t *testing.T, name string) string {
	t.Helper()
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skipf("the go command is not on PATH, so the command cannot be built: %v", err)
	}
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	binary := filepath.Join(t.TempDir(), name)
	build := exec.Command(goTool, "build", "-o", binary, "./cmd/"+name)
	build.Dir = root
	build.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=1")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building %s: %v\n%s", name, err, out)
	}
	return binary
}

// writeManifest saves a manifest beside its model.
func writeManifest(t *testing.T, m onnx.Manifest, dir string) string {
	t.Helper()
	body, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// runCommand runs a built command and returns its standard output, or
// fails with its standard error.
func runCommand(t *testing.T, binary string, args ...string) ([]byte, string, error) {
	t.Helper()
	cmd := exec.Command(binary, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	return stdout.Bytes(), stderr.String(), err
}

func TestTheRewriteCommandVerifiesWritesAndPrintsTheManifest(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	binary := buildCommand(t, "arandu-onnx-rewrite")

	c := col2imCases[0]
	manifest, dir := writeModel(t, syntheticManifest("col2im", "x", 3, "y", 4), c.model())
	manifest.Inputs[0].Shape = []int64{1, 27, 16}
	manifestPath := writeManifest(t, manifest, dir)
	out := filepath.Join(dir, "col2im.convtranspose.onnx")

	stdout, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-library", rt.Config().SharedLibraryPath,
		"-out", out, "-verify", "4")
	if err != nil {
		t.Fatalf("running the command: %v\n%s", err, stderr)
	}
	if !strings.Contains(stderr, "max |original - rewritten| = ") {
		t.Errorf("stderr = %s", stderr)
	}
	printed, err := onnx.LoadManifest(stdout)
	if err != nil {
		t.Fatalf("the printed manifest %s: %v", stdout, err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if printed.File != "col2im.convtranspose.onnx" || printed.SHA256 != sha(body) || printed.Precision != onnx.PrecisionFP32 {
		t.Errorf("printed = %+v, file sha256 %s", printed, sha(body))
	}
	m, err := graph.Load(body)
	if err != nil || m.Graph.OpCounts()["Col2Im"] != 0 {
		t.Errorf("the written model still has Col2Im, or does not load: %v", err)
	}
	session, err := rt.Open(printed, dir)
	if err != nil {
		t.Fatalf("opening the written model with the printed manifest: %v", err)
	}
	_ = session.Close()

	var report struct {
		Rewrite struct {
			Rewritten []struct{ Node string } `json:"rewritten"`
		} `json:"rewrite"`
		Verification struct {
			Inputs     int     `json:"inputs"`
			MaxAbsDiff float64 `json:"max_abs_diff"`
		} `json:"verification"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "col2im.convtranspose.report.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Rewrite.Rewritten) != 1 || report.Verification.Inputs != 4 || report.Verification.MaxAbsDiff > 1e-5 {
		t.Errorf("report = %+v", report)
	}

	// A model whose digest is not its manifest's is refused before it is read.
	tampered := manifest
	tampered.SHA256 = strings.Repeat("0", 64)
	if _, stderr, err := runCommand(t, binary, "-manifest", writeManifest(t, tampered, dir), "-out", filepath.Join(dir, "x.onnx")); err == nil ||
		!strings.Contains(stderr, "digest mismatch") {
		t.Errorf("a tampered model: %v, %s", err, stderr)
	}
}

func TestTheQuantizeCommandExportsFromThePinnedTable(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	binary := buildCommand(t, "arandu-onnx-quantize")

	float := convNet(13)
	table := calibrateConvNet(t, rt, float, "y")
	manifest, dir := writeModel(t, convManifest("y"), float)
	encoded, err := table.MarshalONNXRuntime()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "convnet.table"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest.Calibration = &onnx.Calibration{
		File: "convnet.table", SHA256: sha(encoded), Format: "onnxruntime", Precision: onnx.PrecisionINT8,
		Method: onnx.CalibrationMinMax, Source: "the test network over 32 synthetic inputs",
	}
	manifestPath := writeManifest(t, manifest, dir)
	out := filepath.Join(dir, "convnet.qdq.onnx")

	stdout, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-library", rt.Config().SharedLibraryPath,
		"-mode", "conv-matmul", "-exclude-node", "Gemm_y", "-out", out, "-low", "-1", "-high", "1")
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
	if printed.Precision != onnx.PrecisionINT8 || printed.Quantization != onnx.QuantizationQDQ || printed.Calibration != nil ||
		printed.SHA256 != sha(body) || printed.File != "convnet.qdq.onnx" ||
		printed.DerivedFrom == nil || printed.DerivedFrom.SHA256 != manifest.SHA256 || !strings.Contains(printed.DerivedFrom.Transform, "arandu-onnx-quantize") {
		t.Errorf("printed = %+v", printed)
	}
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := quantize.CheckTensorRT(m); err != nil {
		t.Errorf("CheckTensorRT on the written model: %v", err)
	}
	session, err := rt.Open(printed, dir)
	if err != nil {
		t.Fatalf("opening the export with the printed manifest: %v", err)
	}
	_ = session.Close()

	var report struct {
		Table struct {
			SHA256 string `json:"sha256"`
		} `json:"table"`
		Quantization quantize.Report `json:"quantization"`
		Verification struct {
			Inputs int `json:"inputs"`
		} `json:"verification"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "convnet.qdq.report.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report.Table.SHA256 != sha(encoded) || len(report.Quantization.Quantized) != 2 ||
		len(report.Quantization.Skipped) != 1 || report.Verification.Inputs != 3 {
		t.Errorf("report = %+v", report)
	}

	// A table that is not the one the manifest pins is refused.
	if err := os.WriteFile(filepath.Join(dir, "convnet.table"), append(encoded, 0), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-out", filepath.Join(dir, "other.onnx"), "-verify", "0"); err == nil ||
		!strings.Contains(stderr, "calibration table") {
		t.Errorf("a swapped table: %v, %s", err, stderr)
	}
	// A tolerance the synthetic inputs cannot meet writes nothing.
	if err := os.WriteFile(filepath.Join(dir, "convnet.table"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	again := filepath.Join(dir, "strict.onnx")
	if _, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-table", filepath.Join(dir, "convnet.table"),
		"-out", again, "-tolerance", "1e-9", "-library", rt.Config().SharedLibraryPath); err == nil || !strings.Contains(stderr, "nothing written") {
		t.Errorf("an impossible tolerance: %v, %s", err, stderr)
	}
	if _, err := os.Stat(again); !os.IsNotExist(err) {
		t.Errorf("the refused export was written: %v", err)
	}
}

func TestTheRewriteCommandUpgradesTheOpsetAndFreesTheWeights(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	binary := buildCommand(t, "arandu-onnx-rewrite")

	// An opset 11 network whose weights are graph inputs too, with a
	// Dropout: the three rewrites at once.
	g := message{}.
		sub(1, node("MatMul", []string{"x", "w"}, []string{"xw"})).
		sub(1, node("Dropout", []string{"xw"}, []string{"d"}, floatAttribute("ratio", 0.5))).
		sub(1, node("Unsqueeze", []string{"d"}, []string{"y"}, intsAttribute("axes", 0))).
		text(2, "old-export").
		sub(5, floatInitializer("w", []float32{1, 2, 3, 4, 5, 6}, 3, 2)).
		sub(11, valueInfo("x", protoFloat, -1, 3)).
		sub(11, valueInfo("w", protoFloat, 3, 2)).
		sub(12, valueInfo("y", protoFloat, 1, -1, 2))
	source := syntheticManifest("old-export", "x", 2, "y", 3)
	source.Inputs[0].Shape = []int64{4, 3}
	manifest, dir := writeModel(t, source, modelAt(g, 6, 11))
	manifestPath := writeManifest(t, manifest, dir)
	out := filepath.Join(dir, "old-export.inference.onnx")

	stdout, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-library", rt.Config().SharedLibraryPath,
		"-opset", "13", "-initializers-out-of-inputs", "-dropout-to-identity", "-tolerance", "0", "-out", out)
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
	if printed.SHA256 != sha(body) || printed.DerivedFrom == nil || printed.DerivedFrom.ID != "old-export" ||
		printed.DerivedFrom.SHA256 != manifest.SHA256 || !strings.Contains(printed.DerivedFrom.Transform, "opset 11 to 13") {
		t.Errorf("printed = %+v, derived from %+v", printed, printed.DerivedFrom)
	}
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := m.Opset(graph.DefaultDomain); v != 13 || len(m.Graph.Inputs) != 1 || m.Graph.OpCounts()["Dropout"] != 0 {
		t.Errorf("written at opset %d with %d inputs and ops %v", v, len(m.Graph.Inputs), m.Graph.OpCounts())
	}
	var report struct {
		Initializers struct {
			Removed []string `json:"removed"`
		} `json:"initializers"`
		Dropout struct {
			Replaced []string `json:"replaced"`
		} `json:"dropout"`
		Opset struct {
			OpsetAfter int64 `json:"opset_after"`
		} `json:"opset"`
		Rewrite      *json.RawMessage `json:"rewrite"`
		Verification struct {
			MaxAbsDiff float64 `json:"max_abs_diff"`
		} `json:"verification"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "old-export.inference.report.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Initializers.Removed) != 1 || len(report.Dropout.Replaced) != 1 || report.Opset.OpsetAfter != 13 ||
		report.Rewrite != nil || report.Verification.MaxAbsDiff != 0 {
		t.Errorf("report = %s", raw)
	}

	// Nothing asked for that changes anything is an error, and nothing is
	// written.
	again := filepath.Join(dir, "again.onnx")
	if _, stderr, err := runCommand(t, binary, "-manifest", writeManifest(t, printed, dir), "-opset", "13", "-out", again); err == nil ||
		!strings.Contains(stderr, "changed anything") {
		t.Errorf("a model with nothing to rewrite: %v, %s", err, stderr)
	}
	if _, err := os.Stat(again); !os.IsNotExist(err) {
		t.Errorf("a file was written for a model with nothing to rewrite: %v", err)
	}
}
