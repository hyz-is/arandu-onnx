package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	onnx "github.com/hyz-is/arandu-onnx"
)

// sourceReport names a model file in a report.
type sourceReport struct {
	ID     string `json:"id"`
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
}

// verifyOptions is what the verification reads of the command line.
type verifyOptions struct {
	library, shape string
	verify         int
	seed           uint64
	low, high      float64
}

// verificationReport is what running both models found.
type verificationReport struct {
	Provider        string             `json:"provider"`
	Runtime         string             `json:"runtime"`
	Inputs          int                `json:"inputs"`
	Seed            uint64             `json:"seed"`
	Low             float64            `json:"low"`
	High            float64            `json:"high"`
	MaxAbsDiff      float64            `json:"max_abs_diff"`
	MaxAbsReference float64            `json:"max_abs_reference"`
	PerOutput       map[string]float64 `json:"max_abs_diff_per_output"`
}

// readModel loads the manifest and reads the model it pins, refusing a
// blocked one and a file whose digest is not the manifest's.
func readModel(manifestPath, models string) (onnx.Manifest, string, []byte, error) {
	body, err := os.ReadFile(manifestPath)
	if err != nil {
		return onnx.Manifest{}, "", nil, err
	}
	m, err := onnx.LoadManifest(body)
	if err != nil {
		return onnx.Manifest{}, "", nil, err
	}
	if m.Status == onnx.StatusBlocked {
		return onnx.Manifest{}, "", nil, fmt.Errorf("%w: %s is blocked by its manifest", onnx.ErrModelBlocked, m.ID)
	}
	if models == "" {
		models = filepath.Dir(manifestPath)
	}
	data, err := os.ReadFile(filepath.Join(models, m.File))
	if err != nil {
		return onnx.Manifest{}, "", nil, err
	}
	if got := sha256Hex(data); got != m.SHA256 {
		return onnx.Manifest{}, "", nil, fmt.Errorf("%w: %s has sha256 %s and its manifest records %s", onnx.ErrDigestMismatch, m.File, got, m.SHA256)
	}
	return m, models, data, nil
}

// verifyEquivalent runs the source model and the derived one, each opened
// through its manifest on the CPU, over the same synthetic inputs.
func verifyEquivalent(o verifyOptions, source onnx.Manifest, sourceDir string, derived onnx.Manifest, derivedDir string) (*verificationReport, error) {
	shapes, err := inputShapes(source, o.shape)
	if err != nil {
		return nil, err
	}
	rt, err := onnx.NewRuntime(onnx.Config{SharedLibraryPath: o.library, Providers: []string{onnx.ProviderCPU}})
	if err != nil {
		return nil, err
	}
	if err := rt.Init(); err != nil {
		return nil, fmt.Errorf("verifying needs ONNX Runtime (or -verify 0): %w", err)
	}
	defer func() { _ = rt.Close() }()
	original, err := rt.Open(source, sourceDir)
	if err != nil {
		return nil, err
	}
	defer func() { _ = original.Close() }()
	other, err := rt.Open(derived, derivedDir)
	if err != nil {
		return nil, fmt.Errorf("opening the derived model: %w", err)
	}
	defer func() { _ = other.Close() }()

	report := &verificationReport{
		Provider: onnx.ProviderCPU, Runtime: "onnxruntime " + rt.Version(),
		Inputs: o.verify, Seed: o.seed, Low: o.low, High: o.high, PerOutput: map[string]float64{},
	}
	random := rand.New(rand.NewPCG(o.seed, 0x9e3779b97f4a7c15))
	for range o.verify {
		inputs := map[string]onnx.Tensor{}
		for _, spec := range source.Inputs {
			shape := shapes[spec.Name]
			n := int64(1)
			for _, d := range shape {
				n *= d
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(o.low + (o.high-o.low)*random.Float64())
			}
			inputs[spec.Name] = onnx.Tensor{Shape: shape, Float32: data}
		}
		want, err := original.Run(context.Background(), inputs)
		if err != nil {
			return nil, err
		}
		got, err := other.Run(context.Background(), inputs)
		if err != nil {
			return nil, fmt.Errorf("running the derived model: %w", err)
		}
		for _, spec := range source.Outputs {
			if spec.ElementType != onnx.Float32 {
				continue
			}
			a, b := want.Outputs[spec.Name].Float32, got.Outputs[spec.Name].Float32
			if len(a) != len(b) {
				return nil, fmt.Errorf("output %s has %d values in the original and %d in the derived model", spec.Name, len(a), len(b))
			}
			for i := range a {
				d := math.Abs(float64(a[i]) - float64(b[i]))
				report.MaxAbsDiff = math.Max(report.MaxAbsDiff, d)
				report.PerOutput[spec.Name] = math.Max(report.PerOutput[spec.Name], d)
				report.MaxAbsReference = math.Max(report.MaxAbsReference, math.Abs(float64(a[i])))
			}
		}
	}
	return report, nil
}

// inputShapes is the shape of each synthetic input: the manifest's, or
// -shape for a model with one input whose shape is dynamic.
func inputShapes(m onnx.Manifest, override string) (map[string][]int64, error) {
	out := map[string][]int64{}
	for _, spec := range m.Inputs {
		if spec.ElementType != onnx.Float32 {
			return nil, fmt.Errorf("input %s is %s, and synthetic inputs are float32: pass -verify 0 and verify by hand", spec.Name, spec.ElementType)
		}
		shape := spec.Shape
		if override != "" {
			if len(m.Inputs) != 1 {
				return nil, fmt.Errorf("-shape is for a model with one input, and %s has %d", m.ID, len(m.Inputs))
			}
			shape = nil
			for _, part := range splitList(override) {
				d, err := strconv.ParseInt(part, 10, 64)
				if err != nil || d <= 0 {
					return nil, fmt.Errorf("-shape %q has to be positive integers", override)
				}
				shape = append(shape, d)
			}
		}
		for _, d := range shape {
			if d <= 0 {
				return nil, fmt.Errorf("input %s has shape %v with a dynamic dimension: give -shape", spec.Name, spec.Shape)
			}
		}
		out[spec.Name] = shape
	}
	return out, nil
}

// relativeTo is path relative to dir when it lies inside it, and path
// otherwise.
func relativeTo(dir, path string) string {
	if rel, err := filepath.Rel(dir, path); err == nil && filepath.IsLocal(rel) {
		return filepath.ToSlash(rel)
	}
	return path
}

func writeJSON(path string, v any) error {
	encoded, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(encoded, '\n'), 0o644)
}

func splitList(value string) []string {
	var out []string
	for _, part := range strings.Split(value, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
