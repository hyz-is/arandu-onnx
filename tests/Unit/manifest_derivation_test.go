package unit_test

import (
	"errors"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
)

// derivedJSON is manifestJSON for a file rewritten from another one, as
// arandu-onnx-rewrite prints it.
var derivedJSON = strings.Replace(manifestJSON, `"precision": "fp32"`, `"precision": "fp32",
  "derived_from": {
    "id": "example.source",
    "sha256": "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
    "transform": "arandu-onnx-rewrite: opset 7 to 13, 36 nodes converted (IR 3 to 7)"
  }`, 1)

func TestAManifestRecordsTheFileItWasDerivedFrom(t *testing.T) {
	t.Parallel()

	m, err := onnx.LoadManifest([]byte(derivedJSON))
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if m.DerivedFrom == nil || m.DerivedFrom.ID != "example.source" || !strings.HasPrefix(m.DerivedFrom.Transform, "arandu-onnx-rewrite") {
		t.Fatalf("derived_from = %+v", m.DerivedFrom)
	}
	// A manifest without it is a file as its source published it.
	plain, err := onnx.LoadManifest([]byte(manifestJSON))
	if err != nil || plain.DerivedFrom != nil {
		t.Fatalf("a manifest without derived_from: %+v, %v", plain.DerivedFrom, err)
	}
}

func TestADerivationThatCannotBeFollowedIsRefused(t *testing.T) {
	t.Parallel()

	for name, change := range map[string]func(*onnx.Derivation, onnx.Manifest){
		"no id":              func(d *onnx.Derivation, _ onnx.Manifest) { d.ID = "" },
		"a malformed digest": func(d *onnx.Derivation, _ onnx.Manifest) { d.SHA256 = "abc" },
		"its own digest":     func(d *onnx.Derivation, m onnx.Manifest) { d.SHA256 = m.SHA256 },
		"no transform":       func(d *onnx.Derivation, _ onnx.Manifest) { d.Transform = " " },
	} {
		m, err := onnx.LoadManifest([]byte(derivedJSON))
		if err != nil {
			t.Fatal(err)
		}
		change(m.DerivedFrom, m)
		if err := m.Validate(); !errors.Is(err, onnx.ErrInvalidManifest) || !strings.Contains(err.Error(), "derived_from") {
			t.Errorf("%s: Validate = %v, want ErrInvalidManifest naming derived_from", name, err)
		}
	}
}
