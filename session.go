package onnx

import (
	"context"
	"errors"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// ModelInfo identifies the model and the runtime behind a result.
//
// It is attached to every Result so that an engine records it beside what it
// derived from the outputs: which model, which exact file, which runtime and
// which provider produced the numbers.
type ModelInfo struct {
	// ID is the manifest's model identifier.
	ID string
	// Version is the manifest's model version.
	Version string
	// SHA256 is the digest of the file that was verified and loaded.
	SHA256 string
	// Runtime is the inference runtime and its version, such as
	// "onnxruntime 1.27.1".
	Runtime string
	// Precision is the manifest's numeric precision.
	Precision Precision
	// Provider is the execution provider the session was built for. Nodes
	// that provider does not support run on the CPU inside the same session.
	Provider string
	// Status is the manifest's status when the session was opened.
	Status Status
}

// Result is what one inference returns.
type Result struct {
	// Outputs holds every output the manifest declares, by name.
	Outputs map[string]Tensor
	// Model identifies what produced them.
	Model ModelInfo
}

// Session is one model loaded into ONNX Runtime.
//
// Run is safe for concurrent use. Concurrent calls share the one ONNX Runtime
// session, whose Run is thread-safe, and each call allocates its own input and
// output values, so nothing is shared between them but the read-only graph.
// The session's lock is a reader-writer lock: runs hold it for reading, and
// Close holds it for writing, so Close waits for the runs in flight and no run
// starts on a session being destroyed.
type Session struct {
	runtime  *Runtime
	manifest Manifest
	info     ModelInfo

	mu      sync.RWMutex
	session *ort.DynamicAdvancedSession
}

// Model returns what identifies the session's model and runtime.
func (s *Session) Model() ModelInfo { return s.info }

// Manifest returns the manifest the session was opened with.
func (s *Session) Manifest() Manifest { return cloneManifest(s.manifest) }

// Close destroys the session. It is idempotent and waits for the runs in
// flight.
func (s *Session) Close() error {
	s.mu.Lock()
	session := s.session
	s.session = nil
	s.mu.Unlock()

	if session == nil {
		return nil
	}
	s.runtime.sessionsMu.Lock()
	delete(s.runtime.sessions, s)
	s.runtime.sessionsMu.Unlock()
	return session.Destroy()
}

// Run runs the model on the inputs and returns every output its manifest
// declares, with the ModelInfo attached.
//
// The inputs must be exactly the manifest's, each well formed and conforming
// to its declared element type and shape; anything else is refused with
// ErrInvalidTensor before ONNX Runtime is called.
//
// The context is honoured three times: a context already done is refused
// before anything runs, a context done while the model runs asks ONNX Runtime
// to terminate the run, and a context done by the time the run returns
// discards the outputs and returns the context's error.
func (s *Session) Run(ctx context.Context, inputs map[string]Tensor) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.session == nil {
		return Result{}, ErrClosed
	}

	if err := s.checkInputs(inputs); err != nil {
		return Result{}, err
	}

	values := make([]ort.Value, 0, len(s.manifest.Inputs))
	defer func() {
		for _, value := range values {
			_ = value.Destroy()
		}
	}()
	for _, spec := range s.manifest.Inputs {
		value, err := newValue(inputs[spec.Name])
		if err != nil {
			return Result{}, fmt.Errorf("onnx: %s: input %q: %w", s.info.ID, spec.Name, err)
		}
		values = append(values, value)
	}

	outputs := make([]ort.Value, len(s.manifest.Outputs))
	defer func() {
		for _, value := range outputs {
			if value != nil {
				_ = value.Destroy()
			}
		}
	}()

	if err := s.runWithContext(ctx, values, outputs); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return Result{}, ctxErr
		}
		return Result{}, fmt.Errorf("onnx: %s: run: %w", s.info.ID, err)
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}

	result := Result{Outputs: make(map[string]Tensor, len(outputs)), Model: s.info}
	for i, spec := range s.manifest.Outputs {
		tensor, err := fromValue(outputs[i])
		if err != nil {
			return Result{}, fmt.Errorf("onnx: %s: output %q: %w", s.info.ID, spec.Name, err)
		}
		result.Outputs[spec.Name] = tensor
	}
	return result, nil
}

// runWithContext runs the session with a set of run options that a done
// context terminates.
func (s *Session) runWithContext(ctx context.Context, inputs, outputs []ort.Value) error {
	options, err := ort.NewRunOptions()
	if err != nil {
		return err
	}
	defer options.Destroy()

	// The options must outlive the callback that terminates them: when the
	// callback has started, stop reports false, and the run waits for it to
	// finish before the deferred Destroy frees what it is using.
	terminated := make(chan struct{})
	stop := context.AfterFunc(ctx, func() {
		defer close(terminated)
		_ = options.Terminate()
	})
	defer func() {
		if !stop() {
			<-terminated
		}
	}()

	return s.session.RunWithOptions(inputs, outputs, options)
}

// checkInputs holds the inputs to the manifest's contract.
func (s *Session) checkInputs(inputs map[string]Tensor) error {
	declared := make(map[string]bool, len(s.manifest.Inputs))
	for _, spec := range s.manifest.Inputs {
		declared[spec.Name] = true
		tensor, ok := inputs[spec.Name]
		if !ok {
			return fmt.Errorf("%w: %s: input %q is missing", ErrInvalidTensor, s.info.ID, spec.Name)
		}
		if err := tensor.Validate(); err != nil {
			return fmt.Errorf("%s: input %q: %w", s.info.ID, spec.Name, err)
		}
		if err := tensor.conforms(spec); err != nil {
			return fmt.Errorf("%s: %w", s.info.ID, err)
		}
	}
	for name := range inputs {
		if !declared[name] {
			return fmt.Errorf("%w: %s: input %q is not one the manifest declares", ErrInvalidTensor, s.info.ID, name)
		}
	}
	return nil
}

// newValue wraps a tensor's data in an ONNX Runtime value. The value reads the
// tensor's slice in place for the length of the run.
func newValue(t Tensor) (ort.Value, error) {
	shape := ort.NewShape(t.Shape...)
	switch {
	case t.Float32 != nil:
		return ort.NewTensor(shape, t.Float32)
	case t.Int64 != nil:
		return ort.NewTensor(shape, t.Int64)
	case t.Uint8 != nil:
		return ort.NewTensor(shape, t.Uint8)
	default:
		return nil, ErrInvalidTensor
	}
}

// errUnsupportedOutput is an output whose element type Tensor cannot hold.
var errUnsupportedOutput = errors.New("element type is not float32, int64 or uint8")

// fromValue copies an output ONNX Runtime allocated into a Tensor the caller
// owns.
func fromValue(value ort.Value) (Tensor, error) {
	shape := append([]int64(nil), value.GetShape()...)
	switch typed := value.(type) {
	case *ort.Tensor[float32]:
		return Tensor{Shape: shape, Float32: append([]float32{}, typed.GetData()...)}, nil
	case *ort.Tensor[int64]:
		return Tensor{Shape: shape, Int64: append([]int64{}, typed.GetData()...)}, nil
	case *ort.Tensor[uint8]:
		return Tensor{Shape: shape, Uint8: append([]uint8{}, typed.GetData()...)}, nil
	default:
		return Tensor{}, errUnsupportedOutput
	}
}
