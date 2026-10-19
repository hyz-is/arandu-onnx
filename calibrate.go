package onnx

import (
	"context"
	"fmt"
	"iter"
	"sort"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-onnx/calibration"
	"github.com/hyz-is/arandu-onnx/internal/onnxgraph"
)

// CalibrationResult is what Calibrate measured.
type CalibrationResult struct {
	// Table is the dynamic range of every calibrated tensor, ready to be
	// written with Table.Marshal and pinned in the manifest's calibration
	// entry.
	Table calibration.Table
	// Report is what the collector left out of the table, and why.
	Report calibration.Report
	// Samples is the number of samples each pass read.
	Samples int
	// Excluded is the tensors Options.Exclude and ExcludePrefixes left out,
	// in name order.
	Excluded []string
	// NotFloat is the tensors left out because they are not float32, such as
	// the int64 shape arithmetic a graph carries, in name order. TensorRT
	// takes no range for them.
	NotFloat []string
}

// Samples returns a sample sequence over the inputs given, which Calibrate can
// range over as many times as its method needs.
func Samples(inputs ...map[string]Tensor) iter.Seq2[map[string]Tensor, error] {
	return func(yield func(map[string]Tensor, error) bool) {
		for _, input := range inputs {
			if !yield(input, nil) {
				return
			}
		}
	}
}

// Calibrate runs a float32 model over representative inputs on the CPU and
// returns the INT8 calibration table TensorRT needs to run it in INT8.
//
// The model is verified against its manifest exactly as Open verifies it. The
// verified bytes are then rewritten in memory so that every activation of the
// top-level graph is also a graph output, and that rewritten graph is what
// runs; it is never written to disk and never opened as a session of its own.
// Each sample is held to the manifest's inputs as Session.Run holds them.
//
// samples is ranged over once per pass: once for calibration.MethodMinMax and
// twice for the histogram methods, so it has to yield the same inputs each
// time; Samples builds one from a slice. The inputs decide the table: they
// have to be what the model sees in production, in the same preprocessing.
// Random or synthetic inputs make a table that runs and reads wrong, and are
// for tests only.
//
// The tensors of subgraphs inside control-flow operators are not calibrated.
func (r *Runtime) Calibrate(ctx context.Context, m Manifest, dir string, samples iter.Seq2[map[string]Tensor, error], opts calibration.Options) (CalibrationResult, error) {
	collector, err := calibration.NewCollector(opts)
	if err != nil {
		return CalibrationResult{}, err
	}
	if err := m.Validate(); err != nil {
		return CalibrationResult{}, err
	}
	if m.Status == StatusBlocked {
		return CalibrationResult{}, fmt.Errorf("%w: %s %s is blocked by its manifest", ErrModelBlocked, m.ID, m.Version)
	}
	if m.Precision != PrecisionFP32 {
		return CalibrationResult{}, fmt.Errorf("onnx: %s has precision %s, and calibration runs the float32 model the table is for", m.ID, m.Precision)
	}

	r.mu.RLock()
	defer r.mu.RUnlock()
	if err := r.usable(); err != nil {
		return CalibrationResult{}, err
	}

	data, err := readVerified(m, dir)
	if err != nil {
		return CalibrationResult{}, err
	}
	if err := checkContract(m, data); err != nil {
		return CalibrationResult{}, err
	}

	session, result, inputs, activations, err := r.calibrationSession(m, data, collector.Options())
	if err != nil {
		return CalibrationResult{}, err
	}
	defer func() { _ = session.session.Destroy() }()

	observe := func(record func(string, []float32) error) (int, error) {
		count := 0
		for input, err := range samples {
			if err != nil {
				return count, fmt.Errorf("onnx: calibration sample %d: %w", count, err)
			}
			out, err := session.Run(ctx, input)
			if err != nil {
				return count, fmt.Errorf("onnx: calibration sample %d: %w", count, err)
			}
			for _, name := range inputs {
				if err := record(name, input[name].Float32); err != nil {
					return count, err
				}
			}
			for _, name := range activations {
				if err := record(name, out.Outputs[name].Float32); err != nil {
					return count, err
				}
			}
			count++
		}
		if count == 0 {
			return 0, fmt.Errorf("onnx: calibrating %s: the samples are empty", m.ID)
		}
		return count, nil
	}

	result.Samples, err = observe(collector.ObserveRange)
	if err != nil {
		return CalibrationResult{}, err
	}
	if collector.NeedsHistogram() {
		again, err := observe(collector.ObserveHistogram)
		if err != nil {
			return CalibrationResult{}, err
		}
		if again != result.Samples {
			return CalibrationResult{}, fmt.Errorf("onnx: calibrating %s: the samples yielded %d inputs in the first pass and %d in the second, and the passes have to read the same ones",
				m.ID, result.Samples, again)
		}
	}

	result.Table, result.Report, err = collector.Table()
	if err != nil {
		return CalibrationResult{}, err
	}
	if err := result.Table.Validate(); err != nil {
		return CalibrationResult{}, fmt.Errorf("onnx: calibrating %s: %w", m.ID, err)
	}
	return result, nil
}

// calibrationSession builds the CPU session of the model with every float32
// activation exposed as an output. It returns the session, the result with
// the excluded and non-float tensors filled in, the float32 graph inputs,
// which are observed from the samples themselves, and the float32
// activations, which are observed from the outputs.
func (r *Runtime) calibrationSession(m Manifest, data []byte, opts calibration.Options) (*Session, CalibrationResult, []string, []string, error) {
	var result CalibrationResult
	graph, err := onnxgraph.Read(data)
	if err != nil {
		return nil, result, nil, nil, fmt.Errorf("%w: %s: %v", ErrContractMismatch, m.ID, err)
	}

	isInput := map[string]bool{}
	var inputs []string
	for _, spec := range m.Inputs {
		isInput[spec.Name] = true
		switch {
		case opts.Excluded(spec.Name):
			result.Excluded = append(result.Excluded, spec.Name)
		case spec.ElementType == Float32:
			inputs = append(inputs, spec.Name)
		default:
			result.NotFloat = append(result.NotFloat, spec.Name)
		}
	}

	var wanted []string
	for _, name := range graph.Activations() {
		if isInput[name] {
			continue
		}
		if opts.Excluded(name) {
			result.Excluded = append(result.Excluded, name)
			continue
		}
		wanted = append(wanted, name)
	}
	augmented, err := onnxgraph.WithOutputs(data, wanted)
	if err != nil {
		return nil, result, nil, nil, fmt.Errorf("%w: %s: %v", ErrContractMismatch, m.ID, err)
	}
	_, outputs, err := ort.GetInputOutputInfoWithONNXData(augmented)
	if err != nil {
		return nil, result, nil, nil, fmt.Errorf("onnx: %s: reading the calibration graph: %w", m.ID, err)
	}
	types := make(map[string]ort.InputOutputInfo, len(outputs))
	for _, info := range outputs {
		types[info.Name] = info
	}

	exposed := cloneManifest(m)
	exposed.Outputs = nil
	for _, name := range wanted {
		info, ok := types[name]
		if !ok || info.OrtValueType != ort.ONNXTypeTensor || info.DataType != ort.TensorElementDataTypeFloat {
			result.NotFloat = append(result.NotFloat, name)
			continue
		}
		shape := make([]int64, len(info.Dimensions))
		for i := range shape {
			shape[i] = -1
		}
		exposed.Outputs = append(exposed.Outputs, TensorSpec{Name: name, ElementType: Float32, Shape: shape})
	}
	sort.Strings(result.Excluded)
	sort.Strings(result.NotFloat)
	activations := names(exposed.Outputs)
	if len(exposed.Outputs) == 0 && len(inputs) == 0 {
		return nil, result, nil, nil, fmt.Errorf("onnx: calibrating %s: no float32 tensor is left to calibrate", m.ID)
	}
	if len(exposed.Outputs) == 0 {
		// A session needs an output; the declared ones are fetched and,
		// being excluded or not float, never recorded.
		exposed.Outputs = cloneManifest(m).Outputs
	}

	dynamic, err := r.newSession(exposed, augmented, nil, SessionPlan{}, false)
	if err != nil {
		return nil, result, nil, nil, fmt.Errorf("onnx: creating the calibration session of %s: %w", m.ID, err)
	}
	session := &Session{
		runtime:  r,
		manifest: exposed,
		session:  dynamic,
		info: ModelInfo{
			ID:        m.ID,
			Version:   m.Version,
			SHA256:    m.SHA256,
			Runtime:   "onnxruntime " + r.version,
			Precision: m.Precision,
			Provider:  ProviderCPU,
			Status:    m.Status,
		},
	}
	return session, result, inputs, activations, nil
}
