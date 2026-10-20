package quantize

import (
	"fmt"
	"slices"

	"github.com/hyz-is/arandu-onnx/graph"
)

// A bias stays float32 in the form TensorRT runs, and two things can make it
// otherwise: a DequantizeLinear written in front of it, and ONNX Runtime.
//
// ONNX Runtime's WeightBiasQuantization graph transformer (level 1, the
// level every session gets before its nodes are given to the providers)
// takes a Conv, ConvTranspose or Gemm whose input and weight are read
// through DequantizeLinear, whose bias is a one-dimensional constant and
// whose output goes, through nothing but Relu or Clip and without
// branching, into a QuantizeLinear, and rewrites the bias into an int32
// constant read by a new DequantizeLinear. TensorRT does not dequantize
// int32, so that node runs on the CPU and the TensorRT graph is cut in two
// around every such layer. Its MatMulAddFusion, which runs first, makes a
// Gemm of a MatMul followed by an Add of a one-dimensional constant, so a
// quantized MatMul with a bias is caught the same way. A model in which
// none of this applies is the model TensorRT sees.

// passThrough are the operators a tensor can cross on its way to a bias with
// its values unchanged; the data is their first input.
var passThrough = map[string]bool{
	"Identity": true, "Reshape": true, "Squeeze": true, "Unsqueeze": true,
	"Flatten": true, "Transpose": true, "Cast": true, "Expand": true,
}

// biasUses names every bias the tensors reach, directly or through
// passThrough operators: input 2 of a Conv, ConvTranspose or Gemm, and the
// constant side of an Add that follows a MatMul or a Gemm, which TensorRT
// fuses as its bias.
func biasUses(g *graph.Graph, names []string) []string {
	var uses []string
	seen := map[string]bool{}
	queue := append([]string(nil), names...)
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		for _, c := range g.Consumers(name) {
			if !c.IsStandard() {
				continue
			}
			if i, ok := biasInputs[c.OpType]; ok && c.Input(i) == name {
				uses = append(uses, fmt.Sprintf("%s %q", c.OpType, c.Name))
				continue
			}
			if c.OpType == "Add" && len(c.Inputs) == 2 {
				other := c.Inputs[0]
				if other == name {
					other = c.Inputs[1]
				}
				if p := g.Producer(other); p != nil && (p.OpType == "MatMul" || p.OpType == "Gemm") && p.IsStandard() {
					uses = append(uses, fmt.Sprintf("the Add %q after %s %q", c.Name, p.OpType, p.Name))
					continue
				}
			}
			if passThrough[c.OpType] && c.Input(0) == name {
				queue = append(queue, c.Outputs...)
			}
		}
	}
	return uses
}

// dequantizedFrom returns the DequantizeLinear that writes a tensor, or nil.
func dequantizedFrom(g *graph.Graph, name string) *graph.Node {
	p := g.Producer(name)
	if p == nil || p.OpType != "DequantizeLinear" || !p.IsStandard() {
		return nil
	}
	return p
}

// scalarScale reports whether a QuantizeLinear or DequantizeLinear has one
// constant scale.
func scalarScale(g *graph.Graph, n *graph.Node) bool {
	s := g.Constant(n.Input(1))
	return s != nil && s.Elements() == 1
}

// oneDimensional returns a tensor's constant value when it is a vector.
func oneDimensional(g *graph.Graph, name string) *graph.Tensor {
	t := g.Constant(name)
	if t == nil || len(t.Dims) != 1 {
		return nil
	}
	return t
}

// straightToQuantize reports whether a tensor goes into a QuantizeLinear
// through nothing but Relu and Clip, each step its only consumer, which is
// the path ONNX Runtime's WeightBiasQuantization looks for.
func straightToQuantize(g *graph.Graph, name string) bool {
	for range len(g.Nodes) + 1 {
		consumers := g.Consumers(name)
		if len(consumers) != 1 || !consumers[0].IsStandard() {
			return false
		}
		c := consumers[0]
		switch c.OpType {
		case "QuantizeLinear":
			return true
		case "Relu", "Clip":
			if len(c.Outputs) != 1 {
				return false
			}
			name = c.Outputs[0]
		default:
			return false
		}
	}
	return false
}

// runtimeBias is one node whose bias ONNX Runtime would quantize.
type runtimeBias struct {
	// node is the Conv, ConvTranspose, Gemm or MatMul.
	node *graph.Node
	// add is the Add after a MatMul that carries its bias, or nil.
	add *graph.Node
	// bias is the bias tensor's name.
	bias string
}

// quantizedBiases returns every node whose bias ONNX Runtime's
// WeightBiasQuantization would turn into an int32 DequantizeLinear, after
// its MatMulAddFusion for a MatMul.
func quantizedBiases(g *graph.Graph) []runtimeBias {
	var out []runtimeBias
	for _, n := range g.Nodes {
		if !n.IsStandard() || len(n.Outputs) != 1 {
			continue
		}
		in := dequantizedFrom(g, n.Input(0))
		weight := dequantizedFrom(g, n.Input(1))
		if in == nil || weight == nil || !scalarScale(g, in) {
			continue
		}
		switch n.OpType {
		case "Conv", "ConvTranspose", "Gemm":
			if oneDimensional(g, n.Input(2)) != nil && straightToQuantize(g, n.Outputs[0]) {
				out = append(out, runtimeBias{node: n, bias: n.Input(2)})
			}
		case "MatMul":
			consumers := g.Consumers(n.Outputs[0])
			if len(consumers) != 1 || g.IsOutput(n.Outputs[0]) {
				continue
			}
			add := consumers[0]
			if add.OpType != "Add" || !add.IsStandard() || len(add.Inputs) != 2 || len(add.Outputs) != 1 {
				continue
			}
			bias := add.Inputs[0]
			if bias == n.Outputs[0] {
				bias = add.Inputs[1]
			}
			if oneDimensional(g, bias) != nil && straightToQuantize(g, add.Outputs[0]) {
				out = append(out, runtimeBias{node: n, add: add, bias: bias})
			}
		}
	}
	return out
}

// runtimeBiasQuantization describes, for CheckTensorRT, every node whose bias
// ONNX Runtime would quantize into int32 before TensorRT sees the graph.
func runtimeBiasQuantization(g *graph.Graph) []string {
	var problems []string
	for _, b := range quantizedBiases(g) {
		what := fmt.Sprintf("%s %q", b.node.OpType, b.node.Name)
		if b.add != nil {
			what = fmt.Sprintf("MatMul %q with the bias Add %q", b.node.Name, b.add.Name)
		}
		problems = append(problems, fmt.Sprintf("%s reads dequantized inputs, has the one-dimensional bias %q and goes straight into a QuantizeLinear: ONNX Runtime's WeightBiasQuantization rewrites that bias into an int32 DequantizeLinear, which TensorRT does not run, and the graph is split around it on the CPU; export it again with this version's QDQ, which keeps such biases out of its reach", what, b.bias))
	}
	return problems
}

// detachBiases rewrites every bias ONNX Runtime would quantize so that it
// cannot, keeping what the graph computes: a Conv or ConvTranspose bias
// becomes an Add of the bias shaped [1, C, 1, ...] after the node, and the
// bias of a Gemm, or of the Add after a MatMul, is reshaped from [N] to
// [1, N], which broadcasts the same and which the transformer does not
// take. A MatMul whose activation is not known to have rank 2 or more, and a
// bias that is not float32, are left as they are, for CheckTensorRT to name.
// It returns the biases rewritten.
func detachBiases(g *graph.Graph, namer *graph.Namer) []string {
	var detached []string
	rows := map[string]string{}
	row := func(bias *graph.Tensor) (string, bool) {
		if name, ok := rows[bias.Name]; ok {
			return name, true
		}
		values, err := bias.Float32s()
		if err != nil {
			return "", false
		}
		name := namer.Fresh(bias.Name + "_row")
		g.Initializers = append(g.Initializers, graph.Float32Tensor(name, []int64{1, int64(len(values))}, values))
		rows[bias.Name] = name
		return name, true
	}
	for _, b := range quantizedBiases(g) {
		bias := oneDimensional(g, b.bias)
		switch {
		case b.add != nil:
			t, ok := g.TypeOf(floatSource(g, b.node.Input(0)))
			if !ok || !t.Ranked || len(t.Shape) < 2 {
				continue
			}
			name, ok := row(bias)
			if !ok {
				continue
			}
			for i, in := range b.add.Inputs {
				if in == b.bias {
					b.add.Inputs[i] = name
				}
			}
		case b.node.OpType == "Gemm":
			name, ok := row(bias)
			if !ok {
				continue
			}
			b.node.Inputs[2] = name
		default:
			values, err := bias.Float32s()
			if err != nil {
				continue
			}
			weight := g.Constant(dequantizedFrom(g, b.node.Input(1)).Input(0))
			if weight == nil || len(weight.Dims) < 3 {
				continue
			}
			dims := make([]int64, len(weight.Dims))
			for i := range dims {
				dims[i] = 1
			}
			dims[1] = int64(len(values))
			shaped := namer.Fresh(bias.Name + "_broadcast")
			g.Initializers = append(g.Initializers, graph.Float32Tensor(shaped, dims, values))
			output := b.node.Outputs[0]
			unbiased := namer.Fresh(output + "_without_bias")
			b.node.Inputs = b.node.Inputs[:2]
			b.node.Outputs[0] = unbiased
			add := &graph.Node{
				Name:    namer.Fresh(b.node.Name + "_bias"),
				OpType:  "Add",
				Inputs:  []string{unbiased, shaped},
				Outputs: []string{output},
			}
			at := slices.Index(g.Nodes, b.node)
			g.Nodes = slices.Insert(g.Nodes, at+1, add)
		}
		detached = append(detached, b.bias)
	}
	if !g.HasSubgraphs() {
		for _, name := range detached {
			if g.Initializer(name) != nil && !g.Used(name) && !g.IsInput(name) {
				g.RemoveInitializer(name)
			}
		}
	}
	return detached
}

// floatSource is the float tensor a dequantized activation was quantized
// from: the input of the QuantizeLinear in front of its DequantizeLinear, or
// the name itself.
func floatSource(g *graph.Graph, name string) string {
	if dq := dequantizedFrom(g, name); dq != nil {
		if q := g.Producer(dq.Input(0)); q != nil && q.OpType == "QuantizeLinear" {
			return q.Input(0)
		}
	}
	return name
}
