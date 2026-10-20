// Package rewrite holds graph rewrites that keep a model's numbers and change
// the operators it is written with: so that an execution provider which
// lacks an operator can still run the whole graph (Col2ImToConvTranspose),
// so that an old export reaches the opset quantization needs (UpgradeOpset),
// and so that an export's weights and inference-only operators are what the
// graph says they are (InitializersOutOfInputs, DropoutToIdentity).
//
// Each rewrite works on a graph.Model in memory, is all or nothing (a model
// with one node the rewrite cannot express is refused and left unchanged),
// and returns a report of what it did. Proving that the rewritten model
// computes what the original did is the caller's part: the command
// arandu-onnx-rewrite runs both on ONNX Runtime and prints the difference.
package rewrite

import (
	"errors"
	"fmt"

	"github.com/hyz-is/arandu-onnx/graph"
)

// ErrUnsupported is wrapped by every error that refuses a graph the rewrite
// cannot express.
var ErrUnsupported = errors.New("rewrite: unsupported graph")

// Col2ImOptions completes what a graph does not record.
type Col2ImOptions struct {
	// InputChannels is, by the name of a Col2Im node's first input, the size
	// of that input's dimension 1 (C times the product of block_shape), for a
	// graph whose value info does not record it. A size the graph records is
	// used instead, and a hint that contradicts it is refused.
	InputChannels map[string]int64
}

// Col2ImRewrite is one Col2Im node and what replaced it.
type Col2ImRewrite struct {
	// Node is the Col2Im node's name.
	Node string `json:"node"`
	// Input and Output are its data input and its output, which the
	// ConvTranspose now writes.
	Input  string `json:"input"`
	Output string `json:"output"`
	// Channels is C, the output channels; the ConvTranspose has as many
	// groups.
	Channels int64 `json:"channels"`
	// Image is image_shape and Block is block_shape.
	Image []int64 `json:"image_shape"`
	Block []int64 `json:"block_shape"`
	// Strides, Pads and Dilations are the attributes, defaults filled in.
	Strides   []int64 `json:"strides"`
	Pads      []int64 `json:"pads"`
	Dilations []int64 `json:"dilations"`
	// Blocks is the number of sliding blocks along each spatial dimension;
	// their product is the input's dimension 2.
	Blocks []int64 `json:"blocks"`
	// OutputPadding is what the ConvTranspose adds at the end of each
	// spatial dimension so that its output is image_shape: the rows and
	// columns no block reaches.
	OutputPadding []int64 `json:"output_padding"`
	// Reshape and ConvTranspose are the names of the nodes that replaced
	// it, and Weight the one-hot weight initializer the ConvTranspose reads.
	Reshape       string `json:"reshape"`
	ConvTranspose string `json:"conv_transpose"`
	Weight        string `json:"weight"`
}

// Col2ImReport is what Col2ImToConvTranspose did.
type Col2ImReport struct {
	// Rewritten is every Col2Im node replaced, in graph order.
	Rewritten []Col2ImRewrite `json:"rewritten"`
	// NodesBefore and NodesAfter are the graph's node counts.
	NodesBefore int `json:"nodes_before"`
	NodesAfter  int `json:"nodes_after"`
	// Removed is the initializers and Constant nodes that only the Col2Im
	// nodes read (their image_shape and block_shape), removed with them.
	Removed []string `json:"removed"`
}

// Col2ImToConvTranspose replaces every Col2Im node of the top-level graph
// with a Reshape and a grouped ConvTranspose that compute the same tensor.
//
// Col2Im (opset 18) has no kernel in TensorRT or in ONNX Runtime's CUDA
// provider, so a graph with one is split between the GPU and the CPU. The
// identity used is exact: Col2Im of x [N, C*K, L], with K the product of
// block_shape and L the product of the block counts L1..Lr, is
//
//	Reshape(x, [0, C*K, L1, ..., Lr])
//	ConvTranspose(group = C, kernel_shape = block_shape, the same strides,
//	              pads and dilations, W [C*K, 1, block_shape...])
//
// where W is one-hot: input channel c*K + j has a single 1 at block offset j
// (row-major). Each output pixel is then the sum of the block entries that
// land on it, which is the definition of Col2Im. When image_shape leaves rows
// or columns no block reaches, output_padding adds them as zeros, as Col2Im
// does.
//
// image_shape and block_shape have to be constants (initializers or Constant
// nodes), and the data input's dimension 1 has to be known, from the graph's
// value info or from opts. Anything else is refused with ErrUnsupported and
// the model is left unchanged. Nodes in subgraphs are not rewritten.
func Col2ImToConvTranspose(m *graph.Model, opts Col2ImOptions) (Col2ImReport, error) {
	if m == nil || m.Graph == nil {
		return Col2ImReport{}, fmt.Errorf("%w: the model has no graph", ErrUnsupported)
	}
	g := m.Graph
	report := Col2ImReport{NodesBefore: len(g.Nodes)}

	var plans []col2imPlan
	for _, n := range g.Nodes {
		if n.OpType != "Col2Im" || !n.IsStandard() {
			continue
		}
		p, err := planCol2Im(g, n, opts)
		if err != nil {
			return Col2ImReport{}, err
		}
		plans = append(plans, p)
	}
	if len(plans) == 0 {
		report.NodesAfter = len(g.Nodes)
		return report, nil
	}

	namer := graph.NewNamer(g)
	weights := map[string]string{}
	replaced := map[*graph.Node][]*graph.Node{}
	constants := map[string]bool{}
	for _, p := range plans {
		r := p.rewrite
		k := product(r.Block)

		key := fmt.Sprint(r.Channels, r.Block)
		weight, ok := weights[key]
		if !ok {
			weight = namer.Fresh(r.Node + "/one_hot_weight")
			weights[key] = weight
			g.Initializers = append(g.Initializers, oneHotWeight(weight, r.Channels, r.Block))
		}
		shape := namer.Fresh(r.Node + "/blocks_shape")
		g.Initializers = append(g.Initializers, graph.Int64Tensor(shape, []int64{int64(2 + len(r.Blocks))},
			append([]int64{0, r.Channels * k}, r.Blocks...)))

		blocks := namer.Fresh(r.Input + "/blocks")
		r.Reshape = namer.Fresh(r.Node + "/reshape")
		r.ConvTranspose = namer.Fresh(r.Node + "/conv_transpose")
		r.Weight = weight

		reshape := &graph.Node{
			Name:    r.Reshape,
			OpType:  "Reshape",
			Inputs:  []string{r.Input, shape},
			Outputs: []string{blocks},
		}
		deconv := &graph.Node{
			Name:    r.ConvTranspose,
			OpType:  "ConvTranspose",
			Inputs:  []string{blocks, weight},
			Outputs: []string{r.Output},
			Attributes: []*graph.Attribute{
				graph.IntsAttribute("dilations", r.Dilations),
				graph.IntAttribute("group", r.Channels),
				graph.IntsAttribute("kernel_shape", r.Block),
				graph.IntsAttribute("pads", r.Pads),
				graph.IntsAttribute("strides", r.Strides),
			},
		}
		if anyNonZero(r.OutputPadding) {
			deconv.Attributes = append(deconv.Attributes, graph.IntsAttribute("output_padding", r.OutputPadding))
		}
		replaced[p.node] = []*graph.Node{reshape, deconv}
		constants[p.node.Input(1)] = true
		constants[p.node.Input(2)] = true
		report.Rewritten = append(report.Rewritten, r)
	}

	nodes := make([]*graph.Node, 0, len(g.Nodes)+len(plans))
	for _, n := range g.Nodes {
		if with, ok := replaced[n]; ok {
			nodes = append(nodes, with...)
			continue
		}
		nodes = append(nodes, n)
	}
	g.Nodes = nodes

	if !g.HasSubgraphs() {
		report.Removed = removeUnusedConstants(g, constants)
	}
	if err := g.CheckTopological(); err != nil {
		return Col2ImReport{}, fmt.Errorf("rewrite: the rewritten graph is out of order, which is a defect of the rewrite: %w", err)
	}
	report.NodesAfter = len(g.Nodes)
	return report, nil
}

type col2imPlan struct {
	node    *graph.Node
	rewrite Col2ImRewrite
}

// planCol2Im works out the replacement of one node, or why there is none.
func planCol2Im(g *graph.Graph, n *graph.Node, opts Col2ImOptions) (col2imPlan, error) {
	fail := func(format string, args ...any) (col2imPlan, error) {
		return col2imPlan{}, fmt.Errorf("%w: Col2Im %q: %s", ErrUnsupported, n.Name, fmt.Sprintf(format, args...))
	}
	if len(n.Inputs) != 3 || len(n.Outputs) != 1 || n.Inputs[0] == "" {
		return fail("it has %d inputs and %d outputs, and Col2Im has 3 and 1", len(n.Inputs), len(n.Outputs))
	}
	image, err := constantInts(g, n.Inputs[1])
	if err != nil {
		return fail("image_shape: %v", err)
	}
	block, err := constantInts(g, n.Inputs[2])
	if err != nil {
		return fail("block_shape: %v", err)
	}
	rank := len(image)
	if rank == 0 || len(block) != rank {
		return fail("image_shape %v and block_shape %v have to name the same spatial dimensions", image, block)
	}
	for i := range rank {
		if image[i] <= 0 || block[i] <= 0 {
			return fail("image_shape %v and block_shape %v have to be positive", image, block)
		}
	}
	strides := n.IntsOr("strides", ones(rank))
	dilations := n.IntsOr("dilations", ones(rank))
	pads := n.IntsOr("pads", make([]int64, 2*rank))
	if len(strides) != rank || len(dilations) != rank || len(pads) != 2*rank {
		return fail("strides %v, dilations %v and pads %v do not match %d spatial dimensions", strides, dilations, pads, rank)
	}
	for i := range rank {
		if strides[i] <= 0 || dilations[i] <= 0 || pads[i] < 0 || pads[rank+i] < 0 {
			return fail("strides %v and dilations %v have to be positive and pads %v not negative", strides, dilations, pads)
		}
	}
	for _, a := range n.Attributes {
		switch a.Name {
		case "strides", "dilations", "pads":
		default:
			return fail("attribute %q is not one Col2Im defines", a.Name)
		}
	}

	blocks := make([]int64, rank)
	outputPadding := make([]int64, rank)
	for i := range rank {
		span := image[i] + pads[i] + pads[rank+i] - dilations[i]*(block[i]-1) - 1
		if span < 0 {
			return fail("no block fits along dimension %d: image %d, pads %d+%d, block %d, dilation %d",
				i, image[i], pads[i], pads[rank+i], block[i], dilations[i])
		}
		blocks[i] = span/strides[i] + 1
		outputPadding[i] = span % strides[i]
	}
	k := product(block)

	input := n.Inputs[0]
	recorded := int64(0)
	if t, ok := g.TypeOf(input); ok && t.Ranked {
		if len(t.Shape) != 3 {
			return fail("input %q has rank %d, and Col2Im reads [N, C*K, L]", input, len(t.Shape))
		}
		if t.ElemType != graph.Undefined && t.ElemType != graph.Float {
			// The one-hot weights are float32; a float16 or float64 graph would
			// need them in its own type.
			return fail("input %q is %s, and the one-hot weights are written in float32", input, t.ElemType)
		}
		recorded = t.Shape[1].Value
		if l := t.Shape[2].Value; l > 0 && l != product(blocks) {
			return fail("input %q has %d blocks and image_shape %v with these attributes makes %v = %d",
				input, l, image, blocks, product(blocks))
		}
	}
	hint, hinted := opts.InputChannels[input]
	switch {
	case recorded > 0 && hinted && hint != recorded:
		return fail("input %q has %d channels in the graph and %d in the options", input, recorded, hint)
	case recorded <= 0 && !hinted:
		return fail("the graph does not record dimension 1 of input %q, and the grouped ConvTranspose needs it: give it in Col2ImOptions.InputChannels (arandu-onnx-rewrite -channels tensor=size)", input)
	case recorded <= 0:
		recorded = hint
	}
	if recorded%k != 0 {
		return fail("input %q has %d channels, which is not a multiple of the block size %d", input, recorded, k)
	}
	return col2imPlan{node: n, rewrite: Col2ImRewrite{
		Node:          n.Name,
		Input:         input,
		Output:        n.Outputs[0],
		Channels:      recorded / k,
		Image:         image,
		Block:         block,
		Strides:       strides,
		Pads:          pads,
		Dilations:     dilations,
		Blocks:        blocks,
		OutputPadding: outputPadding,
	}}, nil
}

// constantInts reads a constant int64 vector.
func constantInts(g *graph.Graph, name string) ([]int64, error) {
	t := g.Constant(name)
	if t == nil {
		return nil, fmt.Errorf("%q is computed at run time, and the rewrite needs it constant", name)
	}
	if len(t.Dims) != 1 {
		return nil, fmt.Errorf("%q has shape %v, and has to be a vector", name, t.Dims)
	}
	return t.Int64s()
}

// oneHotWeight is the ConvTranspose weight [C*K, 1, block...] whose input
// channel c*K + j has a single 1 at block offset j.
func oneHotWeight(name string, channels int64, block []int64) *graph.Tensor {
	k := product(block)
	values := make([]float32, channels*k*k)
	for i := range channels * k {
		values[i*k+i%k] = 1
	}
	return graph.Float32Tensor(name, append([]int64{channels * k, 1}, block...), values)
}

// removeUnusedConstants removes the initializers and Constant nodes among
// names that nothing reads any more.
func removeUnusedConstants(g *graph.Graph, names map[string]bool) []string {
	var removed []string
	for _, name := range sortedKeys(names) {
		if name == "" || g.Used(name) || g.IsInput(name) {
			continue
		}
		if g.Initializer(name) != nil {
			g.RemoveInitializer(name)
			removed = append(removed, name)
			continue
		}
		if p := g.Producer(name); p != nil && p.OpType == "Constant" && len(p.Outputs) == 1 {
			kept := g.Nodes[:0]
			for _, n := range g.Nodes {
				if n != p {
					kept = append(kept, n)
				}
			}
			g.Nodes = kept
			removed = append(removed, name)
		}
	}
	return removed
}
