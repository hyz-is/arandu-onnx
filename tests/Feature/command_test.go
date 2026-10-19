package feature_test

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
)

// TestTheCalibrationCommandWritesATableTheManifestCanPin builds
// cmd/arandu-onnx-calibrate, runs it over synthetic samples, and opens the
// model with the entry it printed.
func TestTheCalibrationCommandWritesATableTheManifestCanPin(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skipf("the go command is not on PATH, so the command cannot be built: %v", err)
	}
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	binary := filepath.Join(t.TempDir(), "arandu-onnx-calibrate")
	build := exec.Command(goTool, "build", "-o", binary, "./cmd/arandu-onnx-calibrate")
	build.Dir = root
	build.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=1")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the command: %v\n%s", err, out)
	}

	manifest, dir := writeModel(t, affineManifest(), affineModel())
	body, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(manifestPath, body, 0o600); err != nil {
		t.Fatal(err)
	}

	tablePath := filepath.Join(dir, "affine.calibration")
	run := exec.Command(binary,
		"-manifest", manifestPath,
		"-library", rt.Config().SharedLibraryPath,
		"-synthetic", "16", "-shape", "4,3", "-low", "-1", "-high", "1",
		"-method", "entropy", "-exclude", "y",
		"-out", tablePath)
	var stdout, stderr bytes.Buffer
	run.Stdout, run.Stderr = &stdout, &stderr
	if err := run.Run(); err != nil {
		t.Fatalf("running the command: %v\n%s", err, stderr.String())
	}

	var printed struct {
		Calibration onnx.Calibration `json:"calibration"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &printed); err != nil {
		t.Fatalf("the command printed %q: %v", stdout.String(), err)
	}
	entry := printed.Calibration
	if entry.File != "affine.calibration" || entry.Method != onnx.CalibrationEntropy || entry.Format != calibration.FormatONNXRuntime {
		t.Errorf("printed entry = %+v", entry)
	}
	table, err := os.ReadFile(tablePath)
	if err != nil {
		t.Fatal(err)
	}
	if entry.SHA256 != sha(table) {
		t.Errorf("printed sha256 %s, the file has %s", entry.SHA256, sha(table))
	}
	parsed, err := calibration.Parse(table, calibration.FormatONNXRuntime)
	if err != nil {
		t.Fatalf("parsing the written table: %v", err)
	}
	if _, present := parsed["y"]; present || parsed["x"] <= 0 || parsed["x"] > 1 {
		t.Errorf("table = %v, want x in (0, 1] and y excluded", parsed)
	}

	manifest.Calibration = &entry
	if _, err := rt.Open(manifest, dir); err != nil {
		t.Errorf("opening the model with the printed entry: %v", err)
	}
}
