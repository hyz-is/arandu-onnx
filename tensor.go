package onnx

import (
	"errors"
	"fmt"
)

// Tensor is a dense tensor exchanged with a model: a shape and exactly one
// populated data slice, in row-major order.
//
// It is a plain value so that an engine builds one without this package's
// help and reads one without a type assertion. The slice that is set is the
// element type; the others stay nil.
type Tensor struct {
	// Shape is the dimensions, outermost first.
	Shape []int64
	// Float32 holds the elements of a float32 tensor.
	Float32 []float32
	// Int64 holds the elements of an int64 tensor.
	Int64 []int64
	// Uint8 holds the elements of a uint8 tensor.
	Uint8 []uint8
}

// ErrInvalidTensor is wrapped by every error that reports a tensor which does
// not satisfy its shape or its model's contract.
var ErrInvalidTensor = errors.New("onnx: invalid tensor")

// ElementType reports which data slice is populated. It fails when none or
// more than one is.
func (t Tensor) ElementType() (ElementType, error) {
	var kind ElementType
	set := 0
	if t.Float32 != nil {
		kind, set = Float32, set+1
	}
	if t.Int64 != nil {
		kind, set = Int64, set+1
	}
	if t.Uint8 != nil {
		kind, set = Uint8, set+1
	}
	if set != 1 {
		return "", fmt.Errorf("%w: %d data slices are set, want exactly one", ErrInvalidTensor, set)
	}
	return kind, nil
}

// Len is the number of elements in the populated data slice.
func (t Tensor) Len() int {
	return len(t.Float32) + len(t.Int64) + len(t.Uint8)
}

// Validate reports whether the tensor is well formed: one data slice, every
// dimension positive, and as many elements as the shape holds.
func (t Tensor) Validate() error {
	if _, err := t.ElementType(); err != nil {
		return err
	}
	want := int64(1)
	for _, dim := range t.Shape {
		if dim <= 0 {
			return fmt.Errorf("%w: shape %v has a dimension that is not positive", ErrInvalidTensor, t.Shape)
		}
		if want > int64(^uint(0)>>1)/dim {
			return fmt.Errorf("%w: shape %v holds more elements than a slice can", ErrInvalidTensor, t.Shape)
		}
		want *= dim
	}
	if int64(t.Len()) != want {
		return fmt.Errorf("%w: shape %v holds %d elements and the data has %d", ErrInvalidTensor, t.Shape, want, t.Len())
	}
	return nil
}

// conforms reports whether a well-formed tensor satisfies a spec: the element
// type, the rank, and every dimension the spec fixes.
func (t Tensor) conforms(spec TensorSpec) error {
	kind, err := t.ElementType()
	if err != nil {
		return err
	}
	if kind != spec.ElementType {
		return fmt.Errorf("%w: %q is %s and the model takes %s", ErrInvalidTensor, spec.Name, kind, spec.ElementType)
	}
	if len(t.Shape) != len(spec.Shape) {
		return fmt.Errorf("%w: %q has shape %v and the model takes rank %d %v", ErrInvalidTensor, spec.Name, t.Shape, len(spec.Shape), spec.Shape)
	}
	for i, dim := range spec.Shape {
		if dim != -1 && t.Shape[i] != dim {
			return fmt.Errorf("%w: %q has shape %v and the model takes %v", ErrInvalidTensor, spec.Name, t.Shape, spec.Shape)
		}
	}
	return nil
}
