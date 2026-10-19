package onnx

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/hyz-is/arandu-onnx/calibration"
)

// Status is whether a model may be loaded.
type Status string

// The statuses a Manifest may carry.
const (
	// StatusApproved is a model whose provenance and licence were reviewed and
	// accepted for the uses the manifest records.
	StatusApproved Status = "approved"
	// StatusDevOnly is a model that may be loaded for development and
	// evaluation. It is loaded, and ModelInfo carries the status so the
	// evidence of every result says so.
	StatusDevOnly Status = "dev-only"
	// StatusBlocked is a model that must not run. Open refuses it before it
	// reads the file.
	StatusBlocked Status = "blocked"
)

// CommercialUse is what the model's licence says about commercial use.
type CommercialUse string

// The answers a Manifest may give about commercial use.
const (
	// CommercialUseYes is a licence that allows commercial use.
	CommercialUseYes CommercialUse = "yes"
	// CommercialUseNo is a licence that forbids it.
	CommercialUseNo CommercialUse = "no"
	// CommercialUseUnknown is a licence nobody has settled yet.
	CommercialUseUnknown CommercialUse = "unknown"
)

// Precision is the numeric precision of a model's weights.
type Precision string

// The precisions a Manifest may declare.
const (
	// PrecisionFP32 is 32-bit floating point.
	PrecisionFP32 Precision = "fp32"
	// PrecisionFP16 is 16-bit floating point.
	PrecisionFP16 Precision = "fp16"
	// PrecisionINT8 is 8-bit integer quantization.
	PrecisionINT8 Precision = "int8"
)

// ElementType is the element type of a tensor this package exchanges.
type ElementType string

// The element types a Tensor and a TensorSpec may carry.
const (
	// Float32 is a tensor of float32, held in Tensor.Float32.
	Float32 ElementType = "float32"
	// Int64 is a tensor of int64, held in Tensor.Int64.
	Int64 ElementType = "int64"
	// Uint8 is a tensor of uint8, held in Tensor.Uint8.
	Uint8 ElementType = "uint8"
)

// TensorSpec is one input or output of a model, as its manifest declares it.
type TensorSpec struct {
	// Name is the tensor's name in the graph.
	Name string `json:"name"`
	// ElementType is the type of its elements.
	ElementType ElementType `json:"element_type"`
	// Shape is its dimensions, outermost first. -1 is a dimension the model
	// leaves dynamic.
	Shape []int64 `json:"shape"`
}

// Manifest is the provenance and the contract of one model file.
//
// It is what Open verifies before a session exists: the digest of the bytes,
// the status, and the inputs and outputs the graph must have. The licence
// fields are not checked by code -- they are what a reviewer read before
// setting Status, kept beside the file so the evidence of every result can
// point back to them.
type Manifest struct {
	// ID is the model's stable identifier: lowercase letters, digits, '.',
	// '_' and '-'.
	ID string `json:"id"`
	// Version is the model's version, as its source names it.
	Version string `json:"version"`
	// File is the model file, relative to the directory passed to Open.
	File string `json:"file"`
	// SHA256 is the lowercase hexadecimal SHA-256 of the file.
	SHA256 string `json:"sha256"`
	// SourceURL is where the file was obtained.
	SourceURL string `json:"source_url"`
	// License is the licence of the code that produced the model.
	License string `json:"license"`
	// ModelLicense is the licence of the weights themselves.
	ModelLicense string `json:"model_license"`
	// CommercialUse is what ModelLicense says about commercial use.
	CommercialUse CommercialUse `json:"commercial_use"`
	// Redistribution is what ModelLicense says about shipping the file.
	Redistribution string `json:"redistribution"`
	// Modification is what ModelLicense says about changing the file,
	// including converting or quantizing it.
	Modification string `json:"modification"`
	// Status is whether the model may be loaded.
	Status Status `json:"status"`
	// Inputs is every input of the graph.
	Inputs []TensorSpec `json:"inputs"`
	// Outputs is every output the session returns.
	Outputs []TensorSpec `json:"outputs"`
	// Precision is the numeric precision of the weights.
	Precision Precision `json:"precision"`
	// Calibration is the INT8 calibration table that lets TensorRT run this
	// float model in INT8. Nil for a model without one.
	Calibration *Calibration `json:"calibration,omitempty"`
}

// ErrInvalidManifest is wrapped by every error that reports a manifest which
// cannot be used.
var ErrInvalidManifest = errors.New("onnx: invalid manifest")

var (
	idPattern     = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,127}$`)
	digestPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// LoadManifest decodes a manifest from JSON and validates it.
//
// A field the Manifest type does not have is refused rather than ignored: a
// misspelled key would otherwise be a provenance record that silently lost the
// value somebody wrote.
func LoadManifest(data []byte) (Manifest, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var m Manifest
	if err := decoder.Decode(&m); err != nil {
		return Manifest{}, fmt.Errorf("%w: %v", ErrInvalidManifest, err)
	}
	if decoder.More() {
		return Manifest{}, fmt.Errorf("%w: more than one JSON value", ErrInvalidManifest)
	}
	if err := m.Validate(); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

// Validate reports what makes the manifest unusable. Every error it returns
// wraps ErrInvalidManifest.
//
// The provenance fields are required: a model with no recorded source and
// licence is a model nobody can answer for.
func (m Manifest) Validate() error {
	fail := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s: %s", ErrInvalidManifest, m.ID, fmt.Sprintf(format, args...))
	}

	if !idPattern.MatchString(m.ID) {
		return fmt.Errorf("%w: id %q has to be lowercase letters, digits, '.', '_' and '-', up to 128 characters", ErrInvalidManifest, m.ID)
	}
	if strings.TrimSpace(m.Version) == "" {
		return fail("version is required")
	}
	if m.File == "" || !filepath.IsLocal(m.File) {
		return fail("file %q has to be a relative path inside the model directory", m.File)
	}
	if !digestPattern.MatchString(m.SHA256) {
		return fail("sha256 %q has to be 64 lowercase hexadecimal characters", m.SHA256)
	}
	source, err := url.Parse(m.SourceURL)
	if err != nil || (source.Scheme != "https" && source.Scheme != "http") || source.Host == "" {
		return fail("source_url %q has to be the http(s) address the file was obtained from", m.SourceURL)
	}
	for _, required := range []struct{ field, value string }{
		{"license", m.License},
		{"model_license", m.ModelLicense},
		{"redistribution", m.Redistribution},
		{"modification", m.Modification},
	} {
		if strings.TrimSpace(required.value) == "" {
			return fail("%s is required: provenance is recorded, not assumed", required.field)
		}
	}
	switch m.CommercialUse {
	case CommercialUseYes, CommercialUseNo, CommercialUseUnknown:
	default:
		return fail("commercial_use %q has to be yes, no or unknown", m.CommercialUse)
	}
	switch m.Status {
	case StatusApproved, StatusDevOnly, StatusBlocked:
	default:
		return fail("status %q has to be approved, dev-only or blocked", m.Status)
	}
	switch m.Precision {
	case PrecisionFP32, PrecisionFP16, PrecisionINT8:
	default:
		return fail("precision %q has to be fp32, fp16 or int8", m.Precision)
	}
	if err := validateSpecs("inputs", m.Inputs); err != nil {
		return fail("%v", err)
	}
	if err := validateSpecs("outputs", m.Outputs); err != nil {
		return fail("%v", err)
	}
	if m.Calibration != nil {
		if err := m.Calibration.validate(m); err != nil {
			return fail("%v", err)
		}
	}
	return nil
}

// validateSpecs checks one side of the graph's contract.
func validateSpecs(side string, specs []TensorSpec) error {
	if len(specs) == 0 {
		return fmt.Errorf("%s is empty", side)
	}
	seen := map[string]bool{}
	for _, spec := range specs {
		if spec.Name == "" {
			return fmt.Errorf("%s has a tensor with no name", side)
		}
		if seen[spec.Name] {
			return fmt.Errorf("%s names %q twice", side, spec.Name)
		}
		seen[spec.Name] = true
		switch spec.ElementType {
		case Float32, Int64, Uint8:
		default:
			return fmt.Errorf("%s %q has element_type %q, which is not float32, int64 or uint8", side, spec.Name, spec.ElementType)
		}
		for _, dim := range spec.Shape {
			if dim < -1 || dim == 0 {
				return fmt.Errorf("%s %q has dimension %d: a dimension is positive, or -1 when dynamic", side, spec.Name, dim)
			}
		}
	}
	return nil
}

// names returns the names of the specs, in order.
func names(specs []TensorSpec) []string {
	out := make([]string, len(specs))
	for i, spec := range specs {
		out[i] = spec.Name
	}
	return out
}

// CalibrationMethod is how a calibration table's ranges were chosen, as its
// manifest records it.
type CalibrationMethod string

// The methods a Calibration may record.
const (
	// CalibrationMinMax is calibration.MethodMinMax.
	CalibrationMinMax CalibrationMethod = "minmax"
	// CalibrationPercentile is calibration.MethodPercentile.
	CalibrationPercentile CalibrationMethod = "percentile"
	// CalibrationEntropy is calibration.MethodEntropy.
	CalibrationEntropy CalibrationMethod = "entropy"
	// CalibrationExternal is a table produced by another tool, such as ONNX
	// Runtime's own calibrator; the manifest's source records which.
	CalibrationExternal CalibrationMethod = "external"
)

// Calibration is the INT8 calibration table that travels with a float model,
// pinned like the model is.
//
// Open verifies its SHA-256 and parses it before TensorRT may read it, and
// only a verified copy, written into the TensorRT cache directory, is what
// TensorRT is pointed at.
type Calibration struct {
	// File is the table, relative to the directory passed to Open, like the
	// model's own file.
	File string `json:"file"`
	// SHA256 is the lowercase hexadecimal SHA-256 of the table.
	SHA256 string `json:"sha256"`
	// Format is the table's encoding: calibration.FormatONNXRuntime or
	// calibration.FormatTensorRT.
	Format calibration.Format `json:"format"`
	// Precision is the precision the table calibrates. It is int8, the only
	// one TensorRT takes a table for.
	Precision Precision `json:"precision"`
	// Method is how the ranges were chosen.
	Method CalibrationMethod `json:"method"`
	// Source is what produced the table: the command, the tool and its
	// version, and a description of the samples. Samples that are personal
	// data are described, never named.
	Source string `json:"source"`
}

// validate checks a calibration entry against the manifest that carries it.
func (c *Calibration) validate(m Manifest) error {
	if c.File == "" || !filepath.IsLocal(c.File) {
		return fmt.Errorf("calibration.file %q has to be a relative path inside the model directory", c.File)
	}
	if filepath.Clean(c.File) == filepath.Clean(m.File) {
		return fmt.Errorf("calibration.file %q is the model file itself", c.File)
	}
	if !digestPattern.MatchString(c.SHA256) {
		return fmt.Errorf("calibration.sha256 %q has to be 64 lowercase hexadecimal characters", c.SHA256)
	}
	if !c.Format.Valid() {
		return fmt.Errorf("calibration.format %q has to be %q or %q", c.Format, calibration.FormatONNXRuntime, calibration.FormatTensorRT)
	}
	if c.Precision != PrecisionINT8 {
		return fmt.Errorf("calibration.precision %q has to be int8, the only precision TensorRT takes a table for", c.Precision)
	}
	switch c.Method {
	case CalibrationMinMax, CalibrationPercentile, CalibrationEntropy, CalibrationExternal:
	default:
		return fmt.Errorf("calibration.method %q has to be minmax, percentile, entropy or external", c.Method)
	}
	if strings.TrimSpace(c.Source) == "" {
		return fmt.Errorf("calibration.source is required: provenance is recorded, not assumed")
	}
	if m.Precision != PrecisionFP32 && m.Precision != PrecisionFP16 {
		return fmt.Errorf("a calibration table quantizes a float model, and precision is %q: a model already in int8 carries its scales in the graph", m.Precision)
	}
	return nil
}
