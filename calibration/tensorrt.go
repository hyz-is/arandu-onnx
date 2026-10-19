package calibration

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// DefaultTensorRTHeader is the first line MarshalTensorRT writes when it is
// given none.
//
// ONNX Runtime only checks that the first line contains "TRT-" before any ':'.
// TensorRT itself writes "TRT-<version>-<calibrator>", such as
// "TRT-8601-EntropyCalibration2"; the version 0 and the calibrator name here
// say that no TensorRT calibrator produced the file.
const DefaultTensorRTHeader = "TRT-0-AranduCalibration"

// int8Max is the largest magnitude of a signed 8-bit value, the divisor that
// turns a dynamic range into a TensorRT scale and back.
const int8Max = 127

// MarshalTensorRT encodes the table as TensorRT's calibration cache text: the
// header line, then "name: hex" per tensor in name order, where hex is the
// IEEE-754 bits of range / 127 in lowercase hexadecimal.
//
// A tensor whose name contains ':' is refused, because ONNX Runtime splits
// each line at the first ':' and would read another name. Such a table is
// written with MarshalONNXRuntime. Models exported from TensorFlow name their
// tensors "op:0", so this is not rare.
func (t Table) MarshalTensorRT(header string) ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if header == "" {
		header = DefaultTensorRTHeader
	}
	if before, _, _ := strings.Cut(header, ":"); !strings.Contains(before, "TRT-") || strings.ContainsAny(header, "\n\r") {
		return nil, fmt.Errorf("%w: header %q has to be one line containing \"TRT-\" before any ':'", ErrInvalidTable, header)
	}
	var out bytes.Buffer
	out.WriteString(header)
	out.WriteByte('\n')
	for _, name := range t.Names() {
		if strings.Contains(name, ":") {
			return nil, fmt.Errorf("%w: tensor %q has ':' in its name, which ONNX Runtime reads as the end of the name in TensorRT's format; write the table in the %q format",
				ErrInvalidTable, name, FormatONNXRuntime)
		}
		scale := t[name] / int8Max
		fmt.Fprintf(&out, "%s: %08x\n", name, math.Float32bits(scale))
	}
	return out.Bytes(), nil
}

// ParseTensorRT decodes TensorRT's calibration cache text the way ONNX Runtime
// does: the first line must contain "TRT-" before any ':', and each later line
// is a name, a ':' and the hexadecimal bits of a float32 scale, whose range is
// scale * 127.
//
// It is stricter than ONNX Runtime where ONNX Runtime would read something
// other than what was meant: a line with a second ':' or a scale that is not
// one to eight hexadecimal digits is refused. Empty lines are skipped.
func ParseTensorRT(data []byte) (Table, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	if !scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidTable, err)
		}
		return nil, fmt.Errorf("%w: the TensorRT table is empty", ErrInvalidTable)
	}
	if before, _, _ := strings.Cut(scanner.Text(), ":"); !strings.Contains(before, "TRT-") {
		return nil, fmt.Errorf("%w: the first line %q has no \"TRT-\" header, so ONNX Runtime refuses the table", ErrInvalidTable, scanner.Text())
	}
	t := Table{}
	for line := 2; scanner.Scan(); line++ {
		text := strings.TrimRight(scanner.Text(), "\r")
		if text == "" {
			continue
		}
		parts := strings.Split(text, ":")
		if len(parts) != 2 {
			return nil, fmt.Errorf("%w: line %d %q has %d ':' and ONNX Runtime reads only the text before the first as the name", ErrInvalidTable, line, text, len(parts)-1)
		}
		name, hex := parts[0], strings.TrimSpace(parts[1])
		if len(hex) == 0 || len(hex) > 8 {
			return nil, fmt.Errorf("%w: line %d: scale %q is not one to eight hexadecimal digits", ErrInvalidTable, line, hex)
		}
		bits, err := strconv.ParseUint(hex, 16, 32)
		if err != nil {
			return nil, fmt.Errorf("%w: line %d: scale %q is not hexadecimal", ErrInvalidTable, line, hex)
		}
		if _, dup := t[name]; dup {
			return nil, fmt.Errorf("%w: line %d names %q a second time", ErrInvalidTable, line, name)
		}
		t[name] = math.Float32frombits(uint32(bits)) * int8Max
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidTable, err)
	}
	return t, nil
}
