// Package calibration measures the activation ranges of a float model and
// writes them as the INT8 calibration table the TensorRT execution provider of
// ONNX Runtime reads.
//
// TensorRT runs a float ONNX model in INT8 when it is told, for each tensor,
// the range of values that tensor takes: the "dynamic range". This is implicit
// quantization. The model file stays float; the table carries the ranges; and a
// tensor the table leaves out runs in FP16 or FP32. Leaving tensors out is how
// a head that shares one tensor between pixel boxes and 0-1 scores is kept out
// of INT8, which quantizing it would ruin.
//
// The package is pure Go and calls no runtime: a Collector is fed the values
// each tensor took over a set of representative inputs, and returns a Table.
// Running the model is the job of onnx.Runtime.Calibrate, which feeds it.
//
// # The two file formats
//
// ONNX Runtime's TensorRT provider reads either, chosen by the provider option
// trt_int8_use_native_calibration_table (core/providers/tensorrt/
// tensorrt_execution_provider_utils.h, ReadDynamicRange, identical in ONNX
// Runtime 1.27.0 and 1.30.0):
//
//   - FormatONNXRuntime (the option is 0): a FlatBuffers TrtTable, the schema
//     CalTableFlatBuffers { table KeyValue { key:string (key); value:string; }
//     table TrtTable { dict:[KeyValue]; } }, with each value the dynamic range
//     as decimal text, read with std::stof. It is what ONNX Runtime's own
//     calibrator writes, and the format arandu-alpr runs INT8 with on Tesla
//     T4 cards.
//   - FormatTensorRT (the option is 1): TensorRT's calibration cache text. A
//     first line containing "TRT-", then one "name: hex" line per tensor, where
//     hex is the IEEE-754 bits of the scale, range / 127. ONNX Runtime splits
//     every line at ':', so a tensor whose name contains one cannot be written
//     in this format, and MarshalTensorRT refuses it.
package calibration

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Format is the encoding of a calibration table file.
type Format string

// The formats a table is written in and read from.
const (
	// FormatONNXRuntime is the FlatBuffers table ONNX Runtime's calibrator
	// writes, read with trt_int8_use_native_calibration_table = 0.
	FormatONNXRuntime Format = "onnxruntime"
	// FormatTensorRT is TensorRT's calibration cache text, read with
	// trt_int8_use_native_calibration_table = 1.
	FormatTensorRT Format = "tensorrt"
)

// Valid reports whether the format is one this package reads and writes.
func (f Format) Valid() bool { return f == FormatONNXRuntime || f == FormatTensorRT }

// ErrInvalidTable is wrapped by every error that reports a calibration table
// which cannot be written, read or used.
var ErrInvalidTable = errors.New("calibration: invalid table")

// Table is the dynamic range of each calibrated tensor, by tensor name.
//
// A range r tells TensorRT the tensor's values lie in [-r, r]; it is symmetric
// because TensorRT's INT8 is. A tensor that is absent runs at a higher
// precision.
type Table map[string]float32

// Names returns the tensor names in byte order, the order both formats are
// written in.
func (t Table) Names() []string {
	names := make([]string, 0, len(t))
	for name := range t {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Validate reports a table that TensorRT could not be given: an empty one, a
// tensor with no name or a name with a line break or a NUL, or a range that is
// not a positive finite number.
func (t Table) Validate() error {
	if len(t) == 0 {
		return fmt.Errorf("%w: the table has no tensor", ErrInvalidTable)
	}
	for _, name := range t.Names() {
		if name == "" {
			return fmt.Errorf("%w: a tensor has no name", ErrInvalidTable)
		}
		if strings.ContainsAny(name, "\n\r\x00") {
			return fmt.Errorf("%w: tensor %q has a line break or a NUL in its name", ErrInvalidTable, name)
		}
		r := t[name]
		if math.IsNaN(float64(r)) || math.IsInf(float64(r), 0) || r <= 0 {
			return fmt.Errorf("%w: tensor %q has range %v, and a range is a positive finite number", ErrInvalidTable, name, r)
		}
	}
	return nil
}

// Marshal encodes the table in the format.
func (t Table) Marshal(f Format) ([]byte, error) {
	switch f {
	case FormatONNXRuntime:
		return t.MarshalONNXRuntime()
	case FormatTensorRT:
		return t.MarshalTensorRT("")
	default:
		return nil, fmt.Errorf("%w: format %q is not %q or %q", ErrInvalidTable, f, FormatONNXRuntime, FormatTensorRT)
	}
}

// Parse decodes a table written in the format, and validates it.
func Parse(data []byte, f Format) (Table, error) {
	var (
		t   Table
		err error
	)
	switch f {
	case FormatONNXRuntime:
		t, err = ParseONNXRuntime(data)
	case FormatTensorRT:
		t, err = ParseTensorRT(data)
	default:
		return nil, fmt.Errorf("%w: format %q is not %q or %q", ErrInvalidTable, f, FormatONNXRuntime, FormatTensorRT)
	}
	if err != nil {
		return nil, err
	}
	if err := t.Validate(); err != nil {
		return nil, err
	}
	return t, nil
}

// Detect tells the two formats apart: TensorRT's text starts with a line that
// contains "TRT-", and a FlatBuffers table starts with a binary offset.
func Detect(data []byte) Format {
	first, _, _ := strings.Cut(string(data[:min(len(data), 256)]), "\n")
	if before, _, _ := strings.Cut(first, ":"); strings.Contains(before, "TRT-") {
		return FormatTensorRT
	}
	return FormatONNXRuntime
}
