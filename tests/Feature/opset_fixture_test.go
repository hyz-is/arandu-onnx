package feature_test

import (
	"encoding/binary"
	"math"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/rewrite"
)

// More of the hand-written encoding, for the opset upgrade and the inference
// rewrites: AttributeProto.f = 2 (fixed32) with type 1 (FLOAT), s = 4 with
// type 3 (STRING), floats = 7 (fixed32 each) with type 6 (FLOATS);
// TensorProto.data_type 9 (bool).

const protoBool = 9

// fixed32 writes a 32-bit fixed field.
func (m message) fixed32(field int, v uint32) message {
	return binary.LittleEndian.AppendUint32(m.tag(field, 5), v)
}

// floatAttribute is a FLOAT attribute.
func floatAttribute(name string, v float32) message {
	return message{}.text(1, name).fixed32(2, math.Float32bits(v)).integer(20, 1)
}

// floatsAttribute is a FLOATS attribute.
func floatsAttribute(name string, vs ...float32) message {
	m := message{}.text(1, name)
	for _, v := range vs {
		m = m.fixed32(7, math.Float32bits(v))
	}
	return m.integer(20, 6)
}

// stringAttribute is a STRING attribute.
func stringAttribute(name, s string) message {
	return message{}.text(1, name).text(4, s).integer(20, 3)
}

// boolInitializer is a constant bool scalar stored in the graph.
func boolInitializer(name string, v bool) message {
	raw := []byte{0}
	if v {
		raw[0] = 1
	}
	return message{}.integer(2, protoBool).text(8, name).bytes(9, raw)
}

// opsetCase is a graph of one float input x and one float output y, written
// at an old opset, and the input shape it runs on.
type opsetCase struct {
	name    string
	opset   int64
	ir      int64
	nodes   []message
	inits   []message
	input   []int64
	outRank int
	// converted is the number of nodes the upgrade rewrites.
	converted int
}

func (c opsetCase) model() []byte {
	g := message{}
	for _, n := range c.nodes {
		g = g.sub(1, n)
	}
	g = g.text(2, c.name)
	for _, t := range c.inits {
		g = g.sub(5, t)
	}
	outDims := make([]int64, c.outRank)
	for i := range outDims {
		outDims[i] = -1
	}
	g = g.sub(11, valueInfo("x", protoFloat, c.input...)).sub(12, valueInfo("y", protoFloat, outDims...))
	ir := c.ir
	if ir == 0 {
		ir = 4
	}
	return modelAt(g, ir, c.opset)
}

func (c opsetCase) manifest(id string) onnx.Manifest {
	m := syntheticManifest(id, "x", len(c.input), "y", c.outRank)
	m.Inputs[0].Shape = c.input
	return m
}

// upgraded upgrades a copy of the model to opset 13.
func upgraded(t *testing.T, body []byte) ([]byte, rewrite.OpsetReport) {
	t.Helper()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	report, err := rewrite.UpgradeOpset(m, rewrite.OpsetOptions{})
	if err != nil {
		t.Fatalf("UpgradeOpset: %v", err)
	}
	out, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	return out, report
}

// sameOnCPU runs two models with the same contract on three synthetic inputs
// and returns the largest difference of their outputs.
func sameOnCPU(t *testing.T, rt *onnx.Runtime, a, b onnx.Manifest, aBody, bBody []byte, shape []int64) float64 {
	t.Helper()
	worst := 0.0
	for seed := range uint64(3) {
		input := randomTensor(500+seed, shape...)
		want := runOnce(t, rt, a, aBody, input)
		got := runOnce(t, rt, b, bBody, input)
		worst = math.Max(worst, maxAbsDiff(want, got))
	}
	return worst
}
