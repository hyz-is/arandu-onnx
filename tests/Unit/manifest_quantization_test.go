package unit_test

import (
	"errors"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
)

// qdqJSON is manifestJSON for an int8 QDQ export, as arandu-onnx-quantize
// prints it.
var qdqJSON = strings.Replace(manifestJSON, `"precision": "fp32"`, `"precision": "int8",
  "quantization": "qdq"`, 1)

func TestAManifestDeclaresAQDQModel(t *testing.T) {
	t.Parallel()

	m, err := onnx.LoadManifest([]byte(qdqJSON))
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if m.Precision != onnx.PrecisionINT8 || m.Quantization != onnx.QuantizationQDQ {
		t.Fatalf("precision %q, quantization %q", m.Precision, m.Quantization)
	}
	// A v0.2.0 manifest has no quantization and still loads, an int8 one
	// included.
	plain, err := onnx.LoadManifest([]byte(strings.Replace(manifestJSON, `"fp32"`, `"int8"`, 1)))
	if err != nil || plain.Quantization != "" {
		t.Fatalf("an int8 manifest without quantization: %q, %v", plain.Quantization, err)
	}
}

func TestAQuantizationThatCannotBeUsedIsRefused(t *testing.T) {
	t.Parallel()

	for name, change := range map[string]func(*onnx.Manifest){
		"a float model":        func(m *onnx.Manifest) { m.Precision = onnx.PrecisionFP32 },
		"a half model":         func(m *onnx.Manifest) { m.Precision = onnx.PrecisionFP16 },
		"an unknown form":      func(m *onnx.Manifest) { m.Quantization = "qoperator" },
		"with a table as well": func(m *onnx.Manifest) { m.Calibration = &onnx.Calibration{} },
	} {
		m, err := onnx.LoadManifest([]byte(qdqJSON))
		if err != nil {
			t.Fatal(err)
		}
		change(&m)
		if err := m.Validate(); !errors.Is(err, onnx.ErrInvalidManifest) {
			t.Errorf("%s: Validate = %v, want ErrInvalidManifest", name, err)
		}
	}
}
