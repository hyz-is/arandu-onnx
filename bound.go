package onnx

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-onnx/graph"
)

// BoundSession is a model whose input and output tensors are allocated once,
// when it is opened, and reused by every Run.
//
// Run on a Session builds its input values and copies its outputs on every
// call, which is right for a caller that holds nothing between calls. A
// pipeline that runs one model on one stream of frames wants the opposite: it
// writes the frame into the input's own memory, runs, and reads the outputs in
// place, with nothing allocated per call by this package. Accelerators that
// compile the graph for fixed shapes are used this way.
//
// A BoundSession has one owner. The tensors are shared by every call, so two
// goroutines calling Run, or one writing an input while another runs, corrupt
// each other's data; Close waits for no run. The slices returned by the
// accessors are the tensors' own memory and must never be appended to.
type BoundSession struct {
	session *Session
	dir     string

	quantMu     sync.Mutex
	quantLoaded bool
	quantGraph  *graph.Graph
	quantErr    error

	mu      sync.Mutex
	closed  bool
	inputs  []ort.Value
	outputs []ort.Value

	float32In  map[string][]float32
	uint8In    map[string][]uint8
	int64In    map[string][]int64
	float32Out map[string][]float32
	uint8Out   map[string][]uint8
	int64Out   map[string][]int64
}

// OpenBound is Open for a BoundSession, on Config.Device when the session runs
// on a GPU.
//
// Every dimension of every input and output has to be fixed, except a dynamic
// first dimension, which is bound as 1: the tensors are allocated at their
// shapes. The model is verified and its provider chosen exactly as Open does,
// and Config.Placement applies to it.
func (r *Runtime) OpenBound(m Manifest, dir string) (*BoundSession, error) {
	return r.OpenBoundOn(m, dir, r.cfg.Device)
}

// OpenBoundOn is OpenBound on a given GPU.
func (r *Runtime) OpenBoundOn(m Manifest, dir string, device int) (*BoundSession, error) {
	inputShapes, err := boundShapes(m, "input", m.Inputs)
	if err != nil {
		return nil, err
	}
	outputShapes, err := boundShapes(m, "output", m.Outputs)
	if err != nil {
		return nil, err
	}
	session, err := r.OpenOn(m, dir, device)
	if err != nil {
		return nil, err
	}

	b := &BoundSession{
		session:    session,
		dir:        dir,
		float32In:  map[string][]float32{},
		uint8In:    map[string][]uint8{},
		int64In:    map[string][]int64{},
		float32Out: map[string][]float32{},
		uint8Out:   map[string][]uint8{},
		int64Out:   map[string][]int64{},
	}
	for i, spec := range m.Inputs {
		value, err := b.allocate(spec, inputShapes[i], true)
		if err != nil {
			_ = b.release()
			return nil, fmt.Errorf("onnx: %s: input %q: %w", m.ID, spec.Name, err)
		}
		b.inputs = append(b.inputs, value)
	}
	for i, spec := range m.Outputs {
		value, err := b.allocate(spec, outputShapes[i], false)
		if err != nil {
			_ = b.release()
			return nil, fmt.Errorf("onnx: %s: output %q: %w", m.ID, spec.Name, err)
		}
		b.outputs = append(b.outputs, value)
	}
	return b, nil
}

// boundShapes resolves the shapes a manifest's tensors are allocated at.
func boundShapes(m Manifest, side string, specs []TensorSpec) ([]ort.Shape, error) {
	out := make([]ort.Shape, len(specs))
	for i, spec := range specs {
		shape := append([]int64(nil), spec.Shape...)
		if len(shape) > 0 && shape[0] == -1 {
			shape[0] = 1
		}
		for _, dim := range shape {
			if dim <= 0 {
				return nil, fmt.Errorf("%w: %s: %s %q has shape %v, and a bound session needs every dimension fixed",
					ErrInvalidManifest, m.ID, side, spec.Name, spec.Shape)
			}
		}
		out[i] = ort.NewShape(shape...)
	}
	return out, nil
}

// allocate creates one tensor and records its memory under its name.
func (b *BoundSession) allocate(spec TensorSpec, shape ort.Shape, input bool) (ort.Value, error) {
	switch spec.ElementType {
	case Float32:
		t, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			return nil, err
		}
		if input {
			b.float32In[spec.Name] = t.GetData()
		} else {
			b.float32Out[spec.Name] = t.GetData()
		}
		return t, nil
	case Uint8:
		t, err := ort.NewEmptyTensor[uint8](shape)
		if err != nil {
			return nil, err
		}
		if input {
			b.uint8In[spec.Name] = t.GetData()
		} else {
			b.uint8Out[spec.Name] = t.GetData()
		}
		return t, nil
	case Int64:
		t, err := ort.NewEmptyTensor[int64](shape)
		if err != nil {
			return nil, err
		}
		if input {
			b.int64In[spec.Name] = t.GetData()
		} else {
			b.int64Out[spec.Name] = t.GetData()
		}
		return t, nil
	default:
		return nil, fmt.Errorf("element type %q is not float32, int64 or uint8", spec.ElementType)
	}
}

// InputFloat32 is the memory of a float32 input, or nil when the model has no
// such input.
func (b *BoundSession) InputFloat32(name string) []float32 { return b.float32In[name] }

// InputUint8 is the memory of a uint8 input, or nil when the model has no such
// input.
func (b *BoundSession) InputUint8(name string) []uint8 { return b.uint8In[name] }

// InputInt64 is the memory of an int64 input, or nil when the model has no such
// input.
func (b *BoundSession) InputInt64(name string) []int64 { return b.int64In[name] }

// OutputFloat32 is the memory of a float32 output, overwritten by every Run,
// or nil when the model has no such output.
func (b *BoundSession) OutputFloat32(name string) []float32 { return b.float32Out[name] }

// OutputUint8 is the memory of a uint8 output, overwritten by every Run, or
// nil when the model has no such output.
func (b *BoundSession) OutputUint8(name string) []uint8 { return b.uint8Out[name] }

// OutputInt64 is the memory of an int64 output, overwritten by every Run, or
// nil when the model has no such output.
func (b *BoundSession) OutputInt64(name string) []int64 { return b.int64Out[name] }

// Run runs the model on what the inputs hold and leaves the result in the
// outputs.
func (b *BoundSession) Run() error {
	b.mu.Lock()
	closed := b.closed
	b.mu.Unlock()
	if closed {
		return ErrClosed
	}
	b.session.mu.RLock()
	defer b.session.mu.RUnlock()
	if b.session.session == nil {
		return ErrClosed
	}
	if err := b.session.session.Run(b.inputs, b.outputs); err != nil {
		return fmt.Errorf("onnx: %s: run: %w", b.session.info.ID, err)
	}
	return nil
}

// Model returns what identifies the session's model and runtime.
func (b *BoundSession) Model() ModelInfo { return b.session.Model() }

// Placement returns where ONNX Runtime put the session's nodes. See
// Session.Placement.
func (b *BoundSession) Placement() Placement { return b.session.Placement() }

// Close destroys the session and its tensors. It is idempotent. The memory the
// accessors returned is invalid afterwards.
func (b *BoundSession) Close() error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.closed = true
	b.mu.Unlock()
	return b.release()
}

// release closes the session before the tensors it reads.
//
// A session the runtime already closed, with the library, leaves the tensors
// alone: releasing them would call into a library that is gone. They hold no
// memory ONNX Runtime owns.
func (b *BoundSession) release() error {
	b.session.mu.RLock()
	runtimeClosedIt := b.session.session == nil
	b.session.mu.RUnlock()
	err := b.session.Close()
	if runtimeClosedIt {
		b.inputs, b.outputs = nil, nil
		return err
	}
	for _, v := range b.inputs {
		_ = v.Destroy()
	}
	for _, v := range b.outputs {
		_ = v.Destroy()
	}
	b.inputs, b.outputs = nil, nil
	return err
}
