package unit_test

import (
	"errors"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
)

// calibratedJSON is manifestJSON with a calibration entry, as a model
// directory that ships a table would carry it.
var calibratedJSON = strings.Replace(manifestJSON, `"precision": "fp32"`, `"precision": "fp32",
  "calibration": {
    "file": "det/calibration.flatbuffers",
    "sha256": "60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752",
    "format": "onnxruntime",
    "precision": "int8",
    "method": "minmax",
    "source": "arandu-onnx-calibrate, minmax over 200 document crops"
  }`, 1)

func TestAManifestCarriesItsCalibrationTable(t *testing.T) {
	t.Parallel()

	m, err := onnx.LoadManifest([]byte(calibratedJSON))
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	c := m.Calibration
	if c == nil || c.File != "det/calibration.flatbuffers" || c.Format != calibration.FormatONNXRuntime ||
		c.Precision != onnx.PrecisionINT8 || c.Method != onnx.CalibrationMinMax {
		t.Fatalf("Calibration = %+v", c)
	}

	// A v0.1.0 manifest has none, and still loads.
	plain, err := onnx.LoadManifest([]byte(manifestJSON))
	if err != nil || plain.Calibration != nil {
		t.Fatalf("a manifest without calibration: %+v, %v", plain.Calibration, err)
	}

	misspelled := strings.Replace(calibratedJSON, `"method"`, `"calibrator"`, 1)
	if _, err := onnx.LoadManifest([]byte(misspelled)); !errors.Is(err, onnx.ErrInvalidManifest) {
		t.Errorf("an unknown key inside calibration = %v, want ErrInvalidManifest", err)
	}
}

func TestACalibrationEntryThatCannotBeUsedIsRefused(t *testing.T) {
	t.Parallel()

	for name, change := range map[string]func(*onnx.Manifest){
		"no file":           func(m *onnx.Manifest) { m.Calibration.File = "" },
		"a file outside":    func(m *onnx.Manifest) { m.Calibration.File = "../table" },
		"the model itself":  func(m *onnx.Manifest) { m.Calibration.File = m.File },
		"a short digest":    func(m *onnx.Manifest) { m.Calibration.SHA256 = "60303ae2" },
		"an unknown format": func(m *onnx.Manifest) { m.Calibration.Format = "json" },
		"fp16 precision":    func(m *onnx.Manifest) { m.Calibration.Precision = onnx.PrecisionFP16 },
		"an unknown method": func(m *onnx.Manifest) { m.Calibration.Method = "kl" },
		"no source":         func(m *onnx.Manifest) { m.Calibration.Source = " " },
		"an int8 model":     func(m *onnx.Manifest) { m.Precision = onnx.PrecisionINT8 },
	} {
		m, err := onnx.LoadManifest([]byte(calibratedJSON))
		if err != nil {
			t.Fatal(err)
		}
		change(&m)
		if err := m.Validate(); !errors.Is(err, onnx.ErrInvalidManifest) {
			t.Errorf("%s: Validate = %v, want ErrInvalidManifest", name, err)
		}
	}
}
