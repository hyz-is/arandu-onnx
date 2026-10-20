package rewrite

import "github.com/hyz-is/arandu-onnx/graph"

// inferred is what a rewrite knows of a tensor without running the graph:
// its rank (-1 when unknown), its element type (Undefined when unknown) and,
// for a tensor whose shape is fixed, its dimensions.
type inferred struct {
	rank int
	elem graph.DataType
	dims []int64
}

var unknown = inferred{rank: -1}

// sameAsFirst are the operators whose first output has the rank and the
// element type of their first input.
var sameAsFirst = map[string]bool{
	"Abs": true, "Acos": true, "Acosh": true, "Asin": true, "Asinh": true, "Atan": true, "Atanh": true,
	"AveragePool": true, "BatchNormalization": true, "Ceil": true, "Celu": true, "Clip": true,
	"Conv": true, "ConvTranspose": true, "Cos": true, "Cosh": true, "DepthToSpace": true,
	"Dropout": true, "Elu": true, "Erf": true, "Exp": true, "Floor": true, "GlobalAveragePool": true,
	"GlobalLpPool": true, "GlobalMaxPool": true, "Hardmax": true, "HardSigmoid": true, "HardSwish": true,
	"Identity": true, "InstanceNormalization": true, "LRN": true, "LeakyRelu": true, "Log": true,
	"LogSoftmax": true, "LpPool": true, "MaxPool": true, "Neg": true, "PRelu": true, "Pad": true,
	"Reciprocal": true, "Relu": true, "Resize": true, "Round": true, "Selu": true, "Sigmoid": true,
	"Sign": true, "Sin": true, "Sinh": true, "Slice": true, "Softmax": true, "Softplus": true,
	"Softsign": true, "SpaceToDepth": true, "Split": true, "Sqrt": true, "Tan": true, "Tanh": true,
	"Tile": true, "TopK": true, "Transpose": true, "Upsample": true, "ThresholdedRelu": true,
	"MeanVarianceNormalization": true, "Shrink": true, "CumSum": true,
}

// sameShape are the operators of sameAsFirst whose output also has the
// shape of their input, so a fixed shape carries over.
var sameShape = map[string]bool{
	"Abs": true, "Ceil": true, "Clip": true, "Dropout": true, "Elu": true, "Exp": true, "Floor": true,
	"HardSigmoid": true, "Identity": true, "LeakyRelu": true, "Log": true, "Neg": true, "Reciprocal": true,
	"Relu": true, "Round": true, "Sigmoid": true, "Sign": true, "Sqrt": true, "Tanh": true,
}

// broadcast are the operators whose output has the rank of their largest
// input, under multidirectional broadcasting.
var broadcast = map[string]bool{
	"Add": true, "Sub": true, "Mul": true, "Div": true, "Pow": true, "Max": true, "Min": true,
	"Sum": true, "Mean": true, "Mod": true, "Where": true,
	"And": true, "Or": true, "Xor": true, "Equal": true, "Greater": true, "Less": true,
	"GreaterOrEqual": true, "LessOrEqual": true,
}

// comparisons are the broadcast operators that write bool.
var comparisons = map[string]bool{
	"And": true, "Or": true, "Xor": true, "Equal": true, "Greater": true, "Less": true,
	"GreaterOrEqual": true, "LessOrEqual": true,
}

// infer works out, in graph order, the rank and element type of every tensor
// it can from what the graph records and from the operators' definitions.
// It is a deliberately small inference: what it cannot tell is unknown, and
// a rewrite that needs a value it does not have takes the form that does not
// depend on it, or refuses.
func infer(g *graph.Graph) map[string]inferred {
	out := map[string]inferred{}
	recorded := func(name string) (inferred, bool) {
		t, ok := g.TypeOf(name)
		if !ok {
			return inferred{}, false
		}
		v := inferred{rank: -1, elem: t.ElemType}
		if t.Ranked {
			v.rank = len(t.Shape)
			if dims, ok := t.Static(); ok {
				v.dims = dims
			}
		}
		return v, true
	}
	for _, list := range [][]*graph.ValueInfo{g.Inputs, g.ValueInfo, g.Outputs} {
		for _, vi := range list {
			if v, ok := recorded(vi.Name); ok {
				out[vi.Name] = v
			}
		}
	}
	for _, t := range g.Initializers {
		out[t.Name] = inferred{rank: len(t.Dims), elem: t.DataType, dims: append([]int64{}, t.Dims...)}
	}
	get := func(name string) inferred {
		if v, ok := out[name]; ok {
			return v
		}
		return unknown
	}
	for _, n := range g.Nodes {
		if len(n.Outputs) == 0 {
			continue
		}
		var v inferred
		if n.IsStandard() {
			v = inferNode(g, n, get)
		} else {
			v = unknown
		}
		for i, name := range n.Outputs {
			if name == "" {
				continue
			}
			if r, ok := recorded(name); ok && r.rank >= 0 {
				out[name] = r
				continue
			}
			w := v
			if i > 0 {
				w = secondaryOutput(n, i, v)
			}
			out[name] = w
		}
	}
	return out
}

// secondaryOutput is what infer knows of output i > 0 of a node.
func secondaryOutput(n *graph.Node, i int, first inferred) inferred {
	switch n.OpType {
	case "Split":
		return inferred{rank: first.rank, elem: first.elem}
	case "TopK":
		return inferred{rank: first.rank, elem: graph.Int64}
	}
	return unknown
}

// inferNode is what infer knows of the first output of a standard node.
func inferNode(g *graph.Graph, n *graph.Node, get func(string) inferred) inferred {
	in := get(n.Input(0))
	switch {
	case n.OpType == "Constant":
		if t := g.Constant(n.Outputs[0]); t != nil {
			return inferred{rank: len(t.Dims), elem: t.DataType, dims: append([]int64{}, t.Dims...)}
		}
		return unknown
	case sameShape[n.OpType]:
		return inferred{rank: in.rank, elem: in.elem, dims: in.dims}
	case sameAsFirst[n.OpType]:
		return inferred{rank: in.rank, elem: in.elem}
	case broadcast[n.OpType]:
		rank := 0
		for _, name := range n.Inputs {
			r := get(name).rank
			if r < 0 {
				rank = -1
				break
			}
			rank = max(rank, r)
		}
		elem := in.elem
		if n.OpType == "Where" {
			elem = get(n.Input(1)).elem
		}
		if comparisons[n.OpType] {
			elem = graph.Bool
		}
		return inferred{rank: rank, elem: elem}
	}
	switch n.OpType {
	case "Cast":
		return inferred{rank: in.rank, elem: graph.DataType(n.Int("to", 0)), dims: in.dims}
	case "Shape":
		v := inferred{rank: 1, elem: graph.Int64}
		if in.rank >= 0 && n.Attribute("start") == nil && n.Attribute("end") == nil {
			v.dims = []int64{int64(in.rank)}
		}
		return v
	case "Size":
		return inferred{rank: 0, elem: graph.Int64, dims: []int64{}}
	case "Flatten", "Gemm":
		return inferred{rank: 2, elem: in.elem}
	case "MatMul":
		b := get(n.Input(1))
		switch {
		case in.rank < 0 || b.rank < 0:
			return inferred{rank: -1, elem: in.elem}
		case in.rank >= 2 && b.rank >= 2:
			return inferred{rank: max(in.rank, b.rank), elem: in.elem}
		case in.rank == 1 && b.rank >= 2:
			return inferred{rank: b.rank - 1, elem: in.elem}
		case b.rank == 1 && in.rank >= 2:
			return inferred{rank: in.rank - 1, elem: in.elem}
		}
		return inferred{rank: 0, elem: in.elem}
	case "Reshape", "Expand", "ConstantOfShape":
		shape := get(n.Input(1))
		if n.OpType == "ConstantOfShape" {
			shape = in
		}
		v := inferred{rank: -1, elem: in.elem}
		if n.OpType == "ConstantOfShape" {
			v.elem = graph.Float
			if a := n.Attribute("value"); a != nil && a.T != nil {
				v.elem = a.T.DataType
			}
		}
		if len(shape.dims) == 1 {
			v.rank = int(shape.dims[0])
			if n.OpType == "Expand" && in.rank > v.rank {
				v.rank = in.rank
			}
			if n.OpType == "Expand" && in.rank < 0 {
				v.rank = -1
			}
		}
		return v
	case "Squeeze":
		axes, ok := axesOf(g, n)
		if !ok || in.rank < 0 {
			return inferred{rank: -1, elem: in.elem}
		}
		return inferred{rank: in.rank - len(axes), elem: in.elem}
	case "Unsqueeze":
		axes, ok := axesOf(g, n)
		if !ok || in.rank < 0 {
			return inferred{rank: -1, elem: in.elem}
		}
		return inferred{rank: in.rank + len(axes), elem: in.elem}
	case "Concat":
		for _, name := range n.Inputs {
			if v := get(name); v.rank >= 0 {
				return inferred{rank: v.rank, elem: in.elem}
			}
		}
		return inferred{rank: -1, elem: in.elem}
	case "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
		"ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare":
		if n.Int("keepdims", 1) != 0 {
			return inferred{rank: in.rank, elem: in.elem}
		}
		axes, ok := axesOf(g, n)
		switch {
		case in.rank < 0:
			return inferred{rank: -1, elem: in.elem}
		case !ok:
			return inferred{rank: 0, elem: in.elem}
		}
		return inferred{rank: in.rank - len(axes), elem: in.elem}
	case "ArgMax", "ArgMin":
		rank := in.rank
		if rank >= 0 && n.Int("keepdims", 1) == 0 {
			rank--
		}
		return inferred{rank: rank, elem: graph.Int64}
	case "Gather":
		idx := get(n.Input(1))
		if in.rank < 0 || idx.rank < 0 {
			return inferred{rank: -1, elem: in.elem}
		}
		return inferred{rank: in.rank + idx.rank - 1, elem: in.elem}
	case "Not", "IsNaN", "IsInf":
		return inferred{rank: in.rank, elem: graph.Bool}
	case "NonZero":
		return inferred{rank: 2, elem: graph.Int64}
	case "Range":
		return inferred{rank: 1, elem: in.elem}
	}
	return unknown
}

// axesOf is a node's axes, from its attribute or, from opset 13 on, its
// constant second input; false when it has none or they are computed.
func axesOf(g *graph.Graph, n *graph.Node) ([]int64, bool) {
	if a := n.Attribute("axes"); a != nil {
		return a.Ints, true
	}
	if t := g.Constant(n.Input(1)); t != nil {
		if vs, err := t.Int64s(); err == nil {
			return vs, true
		}
	}
	return nil, false
}
