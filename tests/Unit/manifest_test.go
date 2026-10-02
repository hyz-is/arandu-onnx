package unit_test

import (
	"errors"
	"strings"
	"testing"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// manifestJSON is a complete manifest as a model directory would carry it.
const manifestJSON = `{
  "id": "paddleocr.det",
  "version": "5.0.0",
  "file": "det/model.onnx",
  "sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "source_url": "https://example.com/paddleocr/det.onnx",
  "license": "Apache-2.0",
  "model_license": "Apache-2.0",
  "commercial_use": "yes",
  "redistribution": "allowed with notice",
  "modification": "allowed",
  "status": "approved",
  "inputs": [{"name": "x", "element_type": "float32", "shape": [1, 3, -1, -1]}],
  "outputs": [{"name": "fetch_name_0", "element_type": "float32", "shape": [1, 1, -1, -1]}],
  "precision": "fp32"
}`

func TestAManifestIsReadFromJSON(t *testing.T) {
	t.Parallel()

	m, err := onnx.LoadManifest([]byte(manifestJSON))
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if m.ID != "paddleocr.det" || m.File != "det/model.onnx" || m.Status != onnx.StatusApproved ||
		m.CommercialUse != onnx.CommercialUseYes || m.Precision != onnx.PrecisionFP32 {
		t.Errorf("LoadManifest read %+v", m)
	}
	if got := m.Inputs[0].Shape; len(got) != 4 || got[2] != -1 {
		t.Errorf("input shape = %v, want [1 3 -1 -1]", got)
	}
}

func TestAManifestKeyNobodyDeclaredIsRefused(t *testing.T) {
	t.Parallel()

	misspelled := strings.Replace(manifestJSON, `"model_license"`, `"modelLicense"`, 1)
	if _, err := onnx.LoadManifest([]byte(misspelled)); !errors.Is(err, onnx.ErrInvalidManifest) {
		t.Fatalf("LoadManifest = %v, want ErrInvalidManifest for an unknown key", err)
	}
	if _, err := onnx.LoadManifest([]byte(manifestJSON + manifestJSON)); !errors.Is(err, onnx.ErrInvalidManifest) {
		t.Fatalf("LoadManifest = %v, want ErrInvalidManifest for two documents", err)
	}
}

func TestAManifestWithoutProvenanceIsRefused(t *testing.T) {
	t.Parallel()

	base, err := onnx.LoadManifest([]byte(manifestJSON))
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*onnx.Manifest){
		"no id":                    func(m *onnx.Manifest) { m.ID = "" },
		"an uppercase id":          func(m *onnx.Manifest) { m.ID = "PaddleOCR" },
		"no version":               func(m *onnx.Manifest) { m.Version = " " },
		"no file":                  func(m *onnx.Manifest) { m.File = "" },
		"an absolute file":         func(m *onnx.Manifest) { m.File = "/etc/model.onnx" },
		"a file outside the dir":   func(m *onnx.Manifest) { m.File = "../model.onnx" },
		"a short digest":           func(m *onnx.Manifest) { m.SHA256 = "9f86d081" },
		"an uppercase digest":      func(m *onnx.Manifest) { m.SHA256 = strings.ToUpper(m.SHA256) },
		"no source":                func(m *onnx.Manifest) { m.SourceURL = "" },
		"a source that is no url":  func(m *onnx.Manifest) { m.SourceURL = "somewhere on a drive" },
		"no licence":               func(m *onnx.Manifest) { m.License = "" },
		"no model licence":         func(m *onnx.Manifest) { m.ModelLicense = "" },
		"no redistribution":        func(m *onnx.Manifest) { m.Redistribution = "" },
		"no modification":          func(m *onnx.Manifest) { m.Modification = "" },
		"commercial use maybe":     func(m *onnx.Manifest) { m.CommercialUse = "maybe" },
		"an unknown status":        func(m *onnx.Manifest) { m.Status = "pending" },
		"an unknown precision":     func(m *onnx.Manifest) { m.Precision = "bf16" },
		"no inputs":                func(m *onnx.Manifest) { m.Inputs = nil },
		"no outputs":               func(m *onnx.Manifest) { m.Outputs = nil },
		"an input with no name":    func(m *onnx.Manifest) { m.Inputs[0].Name = "" },
		"an unsupported type":      func(m *onnx.Manifest) { m.Inputs[0].ElementType = "float16" },
		"a zero dimension":         func(m *onnx.Manifest) { m.Inputs[0].Shape = []int64{1, 0} },
		"a dimension below -1":     func(m *onnx.Manifest) { m.Inputs[0].Shape = []int64{-2} },
		"an output named twice":    func(m *onnx.Manifest) { m.Outputs = append(m.Outputs, m.Outputs[0]) },
		"an input with no element": func(m *onnx.Manifest) { m.Inputs[0].ElementType = "" },
	} {
		m := base
		m.Inputs = append([]onnx.TensorSpec(nil), base.Inputs...)
		m.Inputs[0].Shape = append([]int64(nil), base.Inputs[0].Shape...)
		m.Outputs = append([]onnx.TensorSpec(nil), base.Outputs...)
		change(&m)
		if err := m.Validate(); !errors.Is(err, onnx.ErrInvalidManifest) {
			t.Errorf("%s: Validate = %v, want ErrInvalidManifest", name, err)
		}
	}
}

func TestATensorHoldsExactlyWhatItsShapeSays(t *testing.T) {
	t.Parallel()

	good := onnx.Tensor{Shape: []int64{2, 2}, Uint8: []uint8{1, 2, 3, 4}}
	if err := good.Validate(); err != nil {
		t.Fatalf("Validate(%+v) = %v", good, err)
	}
	if kind, _ := good.ElementType(); kind != onnx.Uint8 {
		t.Errorf("ElementType = %q, want uint8", kind)
	}
	scalar := onnx.Tensor{Shape: []int64{}, Int64: []int64{7}}
	if err := scalar.Validate(); err != nil {
		t.Errorf("a scalar is refused: %v", err)
	}

	for name, bad := range map[string]onnx.Tensor{
		"no data":         {Shape: []int64{1}},
		"two kinds":       {Shape: []int64{1}, Float32: []float32{1}, Int64: []int64{1}},
		"too much data":   {Shape: []int64{1}, Float32: []float32{1, 2}},
		"a zero dim":      {Shape: []int64{0}, Float32: []float32{}},
		"a negative dim":  {Shape: []int64{-1}, Float32: []float32{1}},
		"an overflow":     {Shape: []int64{1 << 32, 1 << 32}, Float32: []float32{1}},
		"a scalar of two": {Shape: nil, Float32: []float32{1, 2}},
	} {
		if err := bad.Validate(); !errors.Is(err, onnx.ErrInvalidTensor) {
			t.Errorf("%s: Validate = %v, want ErrInvalidTensor", name, err)
		}
	}
}
