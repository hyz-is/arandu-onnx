package onnx

import (
	"errors"
	"fmt"

	"github.com/hyz-is/arandu-onnx/graph"
)

// QuantParams is the affine quantization of a uint8 or int8 tensor:
// real = Scale * (stored - ZeroPoint).
type QuantParams struct {
	// Scale is the step between two stored values.
	Scale float32
	// ZeroPoint is the stored value that stands for zero.
	ZeroPoint int32
}

// Dequantize returns the real value a stored one stands for.
func (q QuantParams) Dequantize(stored int32) float32 {
	return q.Scale * float32(stored-q.ZeroPoint)
}

// ErrNoQuantization is a tensor whose quantization the graph does not state as
// one scale and one zero point.
var ErrNoQuantization = errors.New("onnx: no per-tensor quantization")

// InputQuantization returns the quantization of a uint8 or int8 input, read
// from the DequantizeLinear node that consumes it in the model. It is
// ErrNoQuantization for an input of another type, one no DequantizeLinear
// reads, or per-channel scales.
func (b *BoundSession) InputQuantization(name string) (QuantParams, error) {
	return b.quantization(name, true)
}

// OutputQuantization returns the quantization of a uint8 or int8 output, read
// from the QuantizeLinear node that produces it in the model, so a caller
// dequantizes with what the model states and not with a number written down
// beside it. The errors are those of InputQuantization.
func (b *BoundSession) OutputQuantization(name string) (QuantParams, error) {
	return b.quantization(name, false)
}

func (b *BoundSession) quantization(name string, input bool) (QuantParams, error) {
	g, err := b.quantizationGraph()
	if err != nil {
		return QuantParams{}, err
	}
	var node *graph.Node
	if input {
		for _, n := range g.Consumers(name) {
			if n.OpType == "DequantizeLinear" && n.IsStandard() {
				node = n
				break
			}
		}
	} else if n := g.Producer(name); n != nil && n.OpType == "QuantizeLinear" && n.IsStandard() {
		node = n
	}
	if node == nil {
		return QuantParams{}, fmt.Errorf("%w: %s: %q is not at the boundary of a QuantizeLinear or DequantizeLinear node", ErrNoQuantization, b.session.info.ID, name)
	}
	return readQuantParams(g, node, b.session.info.ID, name)
}

// quantizationGraph reads the model's graph the first time it is asked for.
func (b *BoundSession) quantizationGraph() (*graph.Graph, error) {
	b.quantMu.Lock()
	defer b.quantMu.Unlock()
	if !b.quantLoaded {
		b.quantGraph, b.quantErr = b.loadGraph()
		b.quantLoaded = true
	}
	return b.quantGraph, b.quantErr
}

// loadGraph reads the verified model again, to read the quantization nodes.
func (b *BoundSession) loadGraph() (*graph.Graph, error) {
	data, err := readVerified(b.session.manifest, b.dir)
	if err != nil {
		return nil, err
	}
	model, err := graph.Load(data)
	if err != nil {
		return nil, fmt.Errorf("onnx: %s: reading the graph: %w", b.session.info.ID, err)
	}
	return model.Graph, nil
}

// readQuantParams reads the scale and zero point of a quantization node.
func readQuantParams(g *graph.Graph, node *graph.Node, id, name string) (QuantParams, error) {
	fail := func(why string) (QuantParams, error) {
		return QuantParams{}, fmt.Errorf("%w: %s: %q: %s", ErrNoQuantization, id, name, why)
	}
	constant := func(tensor string) *graph.Tensor {
		if t := g.Initializer(tensor); t != nil {
			return t
		}
		return g.Constant(tensor)
	}
	scaleTensor := constant(node.Input(1))
	if scaleTensor == nil || scaleTensor.Elements() != 1 {
		return fail("the scale is not a single constant")
	}
	scales, err := scaleTensor.Float32s()
	if err != nil || len(scales) != 1 {
		return fail("the scale cannot be read")
	}
	q := QuantParams{Scale: scales[0]}

	zero := node.Input(2)
	if zero == "" {
		return q, nil // ONNX: absent means 0
	}
	zeroTensor := constant(zero)
	if zeroTensor == nil || zeroTensor.Elements() != 1 {
		return fail("the zero point is not a single constant")
	}
	switch zeroTensor.DataType {
	case graph.Uint8:
		v, err := zeroTensor.Uint8s()
		if err != nil || len(v) != 1 {
			return fail("the zero point cannot be read")
		}
		q.ZeroPoint = int32(v[0])
	case graph.Int8:
		v, err := zeroTensor.Int8s()
		if err != nil || len(v) != 1 {
			return fail("the zero point cannot be read")
		}
		q.ZeroPoint = int32(v[0])
	default:
		return fail("the zero point is not uint8 or int8")
	}
	return q, nil
}
