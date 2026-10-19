package graph

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// DataType is an element type, TensorProto.DataType.
type DataType int32

// The element types of onnx.proto.
const (
	Undefined  DataType = 0
	Float      DataType = 1
	Uint8      DataType = 2
	Int8       DataType = 3
	Uint16     DataType = 4
	Int16      DataType = 5
	Int32      DataType = 6
	Int64      DataType = 7
	String     DataType = 8
	Bool       DataType = 9
	Float16    DataType = 10
	Double     DataType = 11
	Uint32     DataType = 12
	Uint64     DataType = 13
	Complex64  DataType = 14
	Complex128 DataType = 15
	BFloat16   DataType = 16
)

var dataTypeNames = map[DataType]string{
	Undefined: "undefined", Float: "float32", Uint8: "uint8", Int8: "int8",
	Uint16: "uint16", Int16: "int16", Int32: "int32", Int64: "int64",
	String: "string", Bool: "bool", Float16: "float16", Double: "float64",
	Uint32: "uint32", Uint64: "uint64", Complex64: "complex64",
	Complex128: "complex128", BFloat16: "bfloat16",
}

// String names the element type.
func (d DataType) String() string {
	if name, ok := dataTypeNames[d]; ok {
		return name
	}
	return fmt.Sprintf("datatype(%d)", int32(d))
}

// ErrUnreadable is wrapped by the errors of a tensor whose values this
// package cannot read: external data, another element type, or a size that
// does not match the shape.
var ErrUnreadable = errors.New("graph: tensor values unreadable")

// Elements is the number of elements the tensor's shape holds: 1 for a
// scalar.
func (t *Tensor) Elements() int64 {
	n := int64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

func (t *Tensor) readable(want DataType) error {
	if t.External {
		return fmt.Errorf("%w: %s keeps its values in an external data file", ErrUnreadable, t.Name)
	}
	if t.DataType != want {
		return fmt.Errorf("%w: %s is %s, not %s", ErrUnreadable, t.Name, t.DataType, want)
	}
	for _, d := range t.Dims {
		if d < 0 {
			return fmt.Errorf("%w: %s has shape %v", ErrUnreadable, t.Name, t.Dims)
		}
	}
	return nil
}

// Float32s returns the values of a float32 tensor.
func (t *Tensor) Float32s() ([]float32, error) {
	if err := t.readable(Float); err != nil {
		return nil, err
	}
	n := t.Elements()
	if t.RawData != nil || len(t.FloatData) == 0 {
		if int64(len(t.RawData)) != 4*n {
			return nil, fmt.Errorf("%w: %s has %d raw bytes for %d float32 values", ErrUnreadable, t.Name, len(t.RawData), n)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[4*i:]))
		}
		return out, nil
	}
	if int64(len(t.FloatData)) != n {
		return nil, fmt.Errorf("%w: %s has %d float32 values and shape %v", ErrUnreadable, t.Name, len(t.FloatData), t.Dims)
	}
	return append([]float32(nil), t.FloatData...), nil
}

// Int64s returns the values of an int64 tensor.
func (t *Tensor) Int64s() ([]int64, error) {
	if err := t.readable(Int64); err != nil {
		return nil, err
	}
	n := t.Elements()
	if t.RawData != nil || len(t.Int64Data) == 0 {
		if int64(len(t.RawData)) != 8*n {
			return nil, fmt.Errorf("%w: %s has %d raw bytes for %d int64 values", ErrUnreadable, t.Name, len(t.RawData), n)
		}
		out := make([]int64, n)
		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint64(t.RawData[8*i:]))
		}
		return out, nil
	}
	if int64(len(t.Int64Data)) != n {
		return nil, fmt.Errorf("%w: %s has %d int64 values and shape %v", ErrUnreadable, t.Name, len(t.Int64Data), t.Dims)
	}
	return append([]int64(nil), t.Int64Data...), nil
}

// Int8s returns the values of an int8 tensor.
func (t *Tensor) Int8s() ([]int8, error) {
	if err := t.readable(Int8); err != nil {
		return nil, err
	}
	n := t.Elements()
	if t.RawData != nil || len(t.Int32Data) == 0 {
		if int64(len(t.RawData)) != n {
			return nil, fmt.Errorf("%w: %s has %d raw bytes for %d int8 values", ErrUnreadable, t.Name, len(t.RawData), n)
		}
		out := make([]int8, n)
		for i, b := range t.RawData {
			out[i] = int8(b)
		}
		return out, nil
	}
	if int64(len(t.Int32Data)) != n {
		return nil, fmt.Errorf("%w: %s has %d int8 values and shape %v", ErrUnreadable, t.Name, len(t.Int32Data), t.Dims)
	}
	out := make([]int8, n)
	for i, v := range t.Int32Data {
		out[i] = int8(v)
	}
	return out, nil
}

// Float32Tensor returns a float32 tensor, its values stored as raw data.
func Float32Tensor(name string, dims []int64, values []float32) *Tensor {
	raw := make([]byte, 0, 4*len(values))
	for _, v := range values {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
	}
	return &Tensor{Name: name, Dims: append([]int64{}, dims...), DataType: Float, RawData: raw}
}

// Int64Tensor returns an int64 tensor, its values stored as raw data.
func Int64Tensor(name string, dims []int64, values []int64) *Tensor {
	raw := make([]byte, 0, 8*len(values))
	for _, v := range values {
		raw = binary.LittleEndian.AppendUint64(raw, uint64(v))
	}
	return &Tensor{Name: name, Dims: append([]int64{}, dims...), DataType: Int64, RawData: raw}
}

// Int8Tensor returns an int8 tensor, its values stored as raw data.
func Int8Tensor(name string, dims []int64, values []int8) *Tensor {
	raw := make([]byte, len(values))
	for i, v := range values {
		raw[i] = byte(v)
	}
	return &Tensor{Name: name, Dims: append([]int64{}, dims...), DataType: Int8, RawData: raw}
}

// Dim is one dimension of a tensor's shape: a fixed size, or a symbolic name
// or nothing when it is dynamic.
type Dim struct {
	// Value is the size, or 0 when the dimension is not fixed.
	Value int64
	// Param is the symbolic name of a dynamic dimension, such as "batch".
	Param string
}

// TensorType is the type of a tensor value, as a ValueInfo records it.
type TensorType struct {
	// ElemType is the element type.
	ElemType DataType
	// Shape is the dimensions, outermost first. Nil with Ranked false is a
	// tensor whose rank the file does not record.
	Shape []Dim
	// Ranked reports whether the file records the shape at all.
	Ranked bool
}

// Static returns the fixed shape, and false when a dimension is not fixed or
// the rank is unknown.
func (t TensorType) Static() ([]int64, bool) {
	if !t.Ranked {
		return nil, false
	}
	out := make([]int64, len(t.Shape))
	for i, d := range t.Shape {
		if d.Value <= 0 {
			return nil, false
		}
		out[i] = d.Value
	}
	return out, true
}

// The field numbers of TypeProto and the messages under it.
const (
	typeTensor     = 1
	tensorElemType = 1
	tensorShape    = 2
	shapeDim       = 1
	dimValue       = 1
	dimParam       = 2
)

// Tensor returns the tensor type the value info records, and false when it
// records none or another kind of type (a sequence, a map, an optional).
func (v *ValueInfo) Tensor() (TensorType, bool) {
	var out TensorType
	found := false
	err := each(v.typ, func(f field) error {
		if f.num != typeTensor || f.wire != wireBytes {
			return nil
		}
		found = true
		return each(f.value, func(f field) error {
			switch f.num {
			case tensorElemType:
				e, err := f.int()
				out.ElemType = DataType(e)
				return err
			case tensorShape:
				out.Ranked = true
				return each(f.value, func(f field) error {
					if f.num != shapeDim || f.wire != wireBytes {
						return nil
					}
					var d Dim
					err := each(f.value, func(f field) error {
						var err error
						switch f.num {
						case dimValue:
							d.Value, err = f.int()
						case dimParam:
							d.Param, err = f.str()
						}
						return err
					})
					out.Shape = append(out.Shape, d)
					return err
				})
			}
			return nil
		})
	})
	if err != nil || !found {
		return TensorType{}, false
	}
	return out, true
}

// NewValueInfo returns a value info for a tensor of the type given.
func NewValueInfo(name string, t TensorType) *ValueInfo {
	var shape encoder
	for _, d := range t.Shape {
		var dim encoder
		if d.Value > 0 {
			dim.varint(dimValue, uint64(d.Value))
		} else if d.Param != "" {
			dim.str(dimParam, d.Param)
		}
		shape.bytes(shapeDim, dim.encode())
	}
	var tensor encoder
	tensor.varint(tensorElemType, uint64(t.ElemType))
	if t.Ranked || len(t.Shape) > 0 {
		tensor.bytes(tensorShape, shape.encode())
	}
	var typ encoder
	typ.bytes(typeTensor, tensor.encode())
	return &ValueInfo{Name: name, typ: typ.encode()}
}
