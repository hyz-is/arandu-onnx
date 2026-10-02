package feature_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// The models these tests run are written here, byte by byte, as the protobuf
// encoding of an ONNX ModelProto. Nothing generates them: no Python, no
// converter and no file checked in that nobody can read. Each one is small
// enough that its encoding is the clearest statement of what it is.
//
// The field numbers are those of onnx.proto: ModelProto.ir_version = 1,
// graph = 7, opset_import = 8; GraphProto.node = 1, name = 2, initializer = 5,
// input = 11, output = 12; NodeProto.input = 1, output = 2, name = 3,
// op_type = 4, attribute = 5; AttributeProto.name = 1, i = 3, type = 20;
// TensorProto.dims = 1, data_type = 2, name = 8, raw_data = 9;
// ValueInfoProto.name = 1, type = 2; TypeProto.tensor_type = 1;
// TypeProto.Tensor.elem_type = 1, shape = 2; TensorShapeProto.dim = 1;
// Dimension.dim_value = 1, dim_param = 2; OperatorSetIdProto.domain = 1,
// version = 2.

// The TensorProto.DataType values these models use.
const (
	protoFloat = 1
	protoUint8 = 2
	protoInt64 = 7
)

// message is a protobuf message being written.
type message []byte

func (m message) tag(field, wire int) message {
	return m.varint(uint64(field<<3 | wire))
}

func (m message) varint(v uint64) message {
	return binary.AppendUvarint(m, v)
}

// integer writes a varint field.
func (m message) integer(field int, v int64) message {
	return m.tag(field, 0).varint(uint64(v))
}

// bytes writes a length-delimited field.
func (m message) bytes(field int, b []byte) message {
	m = m.tag(field, 2).varint(uint64(len(b)))
	return append(m, b...)
}

func (m message) text(field int, s string) message { return m.bytes(field, []byte(s)) }

func (m message) sub(field int, child message) message { return m.bytes(field, child) }

// valueInfo declares a graph input or output. A dimension of -1 is written as
// a named, dynamic one.
func valueInfo(name string, elemType int, dims ...int64) message {
	var shape message
	for i, d := range dims {
		var dimension message
		if d < 0 {
			dimension = dimension.text(2, "d"+string(rune('0'+i)))
		} else {
			dimension = dimension.integer(1, d)
		}
		shape = shape.sub(1, dimension)
	}
	tensorType := message{}.integer(1, int64(elemType)).sub(2, shape)
	typeProto := message{}.sub(1, tensorType)
	return message{}.text(1, name).sub(2, typeProto)
}

// floatInitializer is a constant float tensor stored in the graph.
func floatInitializer(name string, values []float32, dims ...int64) message {
	var m message
	for _, d := range dims {
		m = m.integer(1, d)
	}
	m = m.integer(2, protoFloat).text(8, name)
	raw := make([]byte, 0, 4*len(values))
	for _, v := range values {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
	}
	return m.bytes(9, raw)
}

// node is one operator of the graph.
func node(op string, inputs, outputs []string, attributes ...message) message {
	var m message
	for _, in := range inputs {
		m = m.text(1, in)
	}
	for _, out := range outputs {
		m = m.text(2, out)
	}
	m = m.text(3, op+"_"+outputs[0]).text(4, op)
	for _, a := range attributes {
		m = m.sub(5, a)
	}
	return m
}

// intAttribute is an INT attribute (AttributeProto.AttributeType 2).
func intAttribute(name string, v int64) message {
	return message{}.text(1, name).integer(3, v).integer(20, 2)
}

// model wraps a graph in a ModelProto at IR version 7 and opset 13.
func model(graph message) []byte {
	opset := message{}.text(1, "").integer(2, 13)
	return message{}.integer(1, 7).text(2, "arandu-onnx-tests").sub(7, graph).sub(8, opset)
}

// affineModel is y = x * w + b followed by label = argmax(y, axis 1).
//
// x is float [N, 3] with a dynamic batch, w is the constant [3, 2]
// {{1, 2}, {3, 4}, {5, 6}}, b is the constant {0.5, -0.5}, y is float [N, 2]
// and label is int64 [N].
func affineModel() []byte {
	graph := message{}.
		sub(1, node("MatMul", []string{"x", "w"}, []string{"xw"})).
		sub(1, node("Add", []string{"xw", "b"}, []string{"y"})).
		sub(1, node("ArgMax", []string{"y"}, []string{"label"}, intAttribute("axis", 1), intAttribute("keepdims", 0))).
		text(2, "affine").
		sub(5, floatInitializer("w", []float32{1, 2, 3, 4, 5, 6}, 3, 2)).
		sub(5, floatInitializer("b", []float32{0.5, -0.5}, 2)).
		sub(11, valueInfo("x", protoFloat, -1, 3)).
		sub(12, valueInfo("y", protoFloat, -1, 2)).
		sub(12, valueInfo("label", protoInt64, -1))
	return model(graph)
}

// castModel is f = float(u), with u uint8 [4] and f float [4].
func castModel() []byte {
	graph := message{}.
		sub(1, node("Cast", []string{"u"}, []string{"f"}, intAttribute("to", protoFloat))).
		text(2, "cast").
		sub(11, valueInfo("u", protoUint8, 4)).
		sub(12, valueInfo("f", protoFloat, 4))
	return model(graph)
}

// affineManifest is the manifest affineModel's file satisfies, with the digest
// filled in by writeModel.
func affineManifest() onnx.Manifest {
	return onnx.Manifest{
		ID:             "test.affine",
		Version:        "1.0.0",
		File:           "affine.onnx",
		SourceURL:      "https://example.com/models/affine.onnx",
		License:        "MIT",
		ModelLicense:   "MIT",
		CommercialUse:  onnx.CommercialUseYes,
		Redistribution: "allowed",
		Modification:   "allowed",
		Status:         onnx.StatusApproved,
		Precision:      onnx.PrecisionFP32,
		Inputs: []onnx.TensorSpec{
			{Name: "x", ElementType: onnx.Float32, Shape: []int64{-1, 3}},
		},
		Outputs: []onnx.TensorSpec{
			{Name: "y", ElementType: onnx.Float32, Shape: []int64{-1, 2}},
			{Name: "label", ElementType: onnx.Int64, Shape: []int64{-1}},
		},
	}
}

// castManifest is the manifest castModel's file satisfies.
func castManifest() onnx.Manifest {
	m := affineManifest()
	m.ID = "test.cast"
	m.File = "cast.onnx"
	m.SourceURL = "https://example.com/models/cast.onnx"
	m.Inputs = []onnx.TensorSpec{{Name: "u", ElementType: onnx.Uint8, Shape: []int64{4}}}
	m.Outputs = []onnx.TensorSpec{{Name: "f", ElementType: onnx.Float32, Shape: []int64{4}}}
	return m
}

// writeModel saves a model under a fresh directory and records its digest in
// the manifest.
func writeModel(t *testing.T, m onnx.Manifest, body []byte) (onnx.Manifest, string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, m.File), body, 0o600); err != nil {
		t.Fatalf("writing the model: %v", err)
	}
	sum := sha256.Sum256(body)
	m.SHA256 = hex.EncodeToString(sum[:])
	return m, dir
}

// newRuntime returns an initialised runtime on the default library, closed when
// the test ends, or skips the test when the library is not on this machine.
func newRuntime(t *testing.T, cfg onnx.Config) *onnx.Runtime {
	t.Helper()
	runtime, err := onnx.NewRuntime(cfg)
	if err != nil {
		t.Fatalf("building the runtime: %v", err)
	}
	if err := runtime.Init(); err != nil {
		if errors.Is(err, onnx.ErrLibraryUnavailable) {
			t.Skipf("ONNX Runtime is not available here, so nothing that runs a model can be tested: %v", err)
		}
		t.Fatalf("initialising the runtime: %v", err)
	}
	t.Cleanup(func() {
		if err := runtime.Close(); err != nil {
			t.Errorf("closing the runtime: %v", err)
		}
	})
	return runtime
}
