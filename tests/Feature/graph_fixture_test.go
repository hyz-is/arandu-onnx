package feature_test

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
)

// More of the hand-written encoding, for the models the rewrite and
// quantization tests run: AttributeProto.ints = 8 with type 7 (INTS);
// TensorProto.data_type 7 (int64) with raw_data.

// intsAttribute is an INTS attribute.
func intsAttribute(name string, vs ...int64) message {
	m := message{}.text(1, name)
	for _, v := range vs {
		m = m.integer(8, v)
	}
	return m.integer(20, 7)
}

// int64Initializer is a constant int64 tensor stored in the graph.
func int64Initializer(name string, values []int64, dims ...int64) message {
	var m message
	for _, d := range dims {
		m = m.integer(1, d)
	}
	m = m.integer(2, protoInt64).text(8, name)
	raw := make([]byte, 0, 8*len(values))
	for _, v := range values {
		raw = binary.LittleEndian.AppendUint64(raw, uint64(v))
	}
	return m.bytes(9, raw)
}

// modelAt wraps a graph in a ModelProto at an IR version and an opset.
func modelAt(graph message, ir, opset int64) []byte {
	opsetImport := message{}.text(1, "").integer(2, opset)
	return message{}.integer(1, ir).text(2, "arandu-onnx-tests").sub(7, graph).sub(8, opsetImport)
}

// syntheticManifest is the manifest of a test model with one float input and
// one float output of the given ranks, every dimension dynamic.
func syntheticManifest(id, input string, inputRank int, output string, outputRank int) onnx.Manifest {
	m := affineManifest()
	m.ID = id
	m.File = id + ".onnx"
	m.SourceURL = "https://example.com/models/" + id + ".onnx"
	dynamic := func(rank int) []int64 {
		out := make([]int64, rank)
		for i := range out {
			out[i] = -1
		}
		return out
	}
	m.Inputs = []onnx.TensorSpec{{Name: input, ElementType: onnx.Float32, Shape: dynamic(inputRank)}}
	m.Outputs = []onnx.TensorSpec{{Name: output, ElementType: onnx.Float32, Shape: dynamic(outputRank)}}
	return m
}

// runOnce opens a model with its manifest on the CPU and runs it once.
func runOnce(t *testing.T, rt *onnx.Runtime, m onnx.Manifest, body []byte, input onnx.Tensor) onnx.Tensor {
	t.Helper()
	m, dir := writeModel(t, m, body)
	session, err := rt.Open(m, dir)
	if err != nil {
		t.Fatalf("opening %s: %v", m.ID, err)
	}
	defer func() { _ = session.Close() }()
	result, err := session.Run(context.Background(), map[string]onnx.Tensor{m.Inputs[0].Name: input})
	if err != nil {
		t.Fatalf("running %s: %v", m.ID, err)
	}
	return result.Outputs[m.Outputs[0].Name]
}

// randomTensor is a tensor of uniform values in [-1, 1), the same for a seed.
func randomTensor(seed uint64, shape ...int64) onnx.Tensor {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	random := rand.New(rand.NewPCG(seed, 0x2545f4914f6cdd1d))
	data := make([]float32, n)
	for i := range data {
		data[i] = 2*random.Float32() - 1
	}
	return onnx.Tensor{Shape: shape, Float32: data}
}

// maxAbsDiff is the largest |a - b|, and +Inf when the tensors differ in
// shape.
func maxAbsDiff(a, b onnx.Tensor) float64 {
	if len(a.Float32) != len(b.Float32) || len(a.Shape) != len(b.Shape) {
		return math.Inf(1)
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return math.Inf(1)
		}
	}
	worst := 0.0
	for i := range a.Float32 {
		worst = math.Max(worst, math.Abs(float64(a.Float32[i])-float64(b.Float32[i])))
	}
	return worst
}
