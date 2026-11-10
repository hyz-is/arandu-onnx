package rewrite

import (
	"fmt"

	"github.com/hyz-is/arandu-onnx/graph"
)

// AveragePoolOptions completes what a graph does not record.
type AveragePoolOptions struct {
	// InputChannels is, by the name of an AveragePool node's input, the size
	// of that input's dimension 1, for a graph whose value info does not
	// record it. A size the graph records is used instead, and a hint that
	// contradicts it is refused.
	InputChannels map[string]int64
}

// AveragePoolRewrite is one AveragePool node and what replaced it.
type AveragePoolRewrite struct {
	// Node is the AveragePool node's name, and Conv the name of the Conv
	// that now writes its output.
	Node string `json:"node"`
	Conv string `json:"conv"`
	// Input and Output are the tensors it reads and writes, unchanged.
	Input  string `json:"input"`
	Output string `json:"output"`
	// Channels is C: the Conv has as many groups.
	Channels int64 `json:"channels"`
	// Kernel and Strides are the pool's attributes, carried over.
	Kernel  []int64 `json:"kernel_shape"`
	Strides []int64 `json:"strides"`
	// Weight is the constant [C, 1, kernel...] filled with 1 / product(kernel).
	Weight string `json:"weight"`
}

// AveragePoolReport is what AveragePoolToDepthwiseConv did.
type AveragePoolReport struct {
	// Rewritten is every AveragePool node replaced, in graph order.
	Rewritten []AveragePoolRewrite `json:"rewritten"`
	// NodesBefore and NodesAfter are the graph's node counts.
	NodesBefore int `json:"nodes_before"`
	NodesAfter  int `json:"nodes_after"`
}

// AveragePoolToDepthwiseConv replaces every AveragePool node of the top-level
// graph with a depthwise Conv that computes the same tensor.
//
// An average over a k1 x ... x kr window with no padding is the convolution of
// each channel with a constant kernel of 1 / (k1 * ... * kr): Conv with
// group = C, kernel_shape and strides of the pool, no pads, and a weight
// [C, 1, k1, ..., kr] holding that constant. The two agree to rounding. It
// matters on accelerators where a quantized pooling is far slower than a
// convolution of the same size (a 2 x 2, stride 1 pool measured at hundreds
// of times the cost of the depthwise convolution on one NPU); on others the
// two cost the same.
//
// Only the pool whose result the identity gives exactly is rewritten: float32
// data, every pad 0, ceil_mode 0, dilations 1, auto_pad NOTSET or VALID, and
// the input's dimension 1 (C) known, from the graph's value info or from
// opts. Any other AveragePool node is refused with ErrUnsupported and the
// model is left unchanged, as is a model with no AveragePool. The Conv keeps
// the pool's input and output tensor names. Nodes in subgraphs are not
// rewritten.
func AveragePoolToDepthwiseConv(m *graph.Model, opts AveragePoolOptions) (AveragePoolReport, error) {
	if m == nil || m.Graph == nil {
		return AveragePoolReport{}, fmt.Errorf("%w: the model has no graph", ErrUnsupported)
	}
	g := m.Graph
	report := AveragePoolReport{NodesBefore: len(g.Nodes)}

	type plan struct {
		node     *graph.Node
		channels int64
		kernel   []int64
		strides  []int64
	}
	var plans []plan
	for _, n := range g.Nodes {
		if n.OpType != "AveragePool" || !n.IsStandard() {
			continue
		}
		fail := func(format string, args ...any) error {
			return fmt.Errorf("%w: AveragePool %q: %s", ErrUnsupported, n.Name, fmt.Sprintf(format, args...))
		}
		if len(n.Inputs) != 1 || len(n.Outputs) != 1 || n.Inputs[0] == "" || n.Outputs[0] == "" {
			return AveragePoolReport{}, fail("it has to read one tensor and write one")
		}
		kernel := n.IntsOr("kernel_shape", nil)
		if len(kernel) == 0 {
			return AveragePoolReport{}, fail("it has no kernel_shape")
		}
		for _, k := range kernel {
			if k <= 0 {
				return AveragePoolReport{}, fail("kernel_shape %v has a size that is not positive", kernel)
			}
		}
		if anyNonZero(n.IntsOr("pads", nil)) {
			return AveragePoolReport{}, fail("it pads, and the average then depends on count_include_pad at the border")
		}
		if n.Int("ceil_mode", 0) != 0 {
			return AveragePoolReport{}, fail("ceil_mode is set, which adds partial windows")
		}
		for _, d := range n.IntsOr("dilations", nil) {
			if d != 1 {
				return AveragePoolReport{}, fail("dilations %v are not 1", n.IntsOr("dilations", nil))
			}
		}
		if a := n.Attribute("auto_pad"); a != nil && string(a.S) != "" && string(a.S) != "NOTSET" && string(a.S) != "VALID" {
			return AveragePoolReport{}, fail("auto_pad %q pads", string(a.S))
		}
		strides := n.IntsOr("strides", ones(len(kernel)))
		if len(strides) != len(kernel) {
			return AveragePoolReport{}, fail("strides %v do not match kernel_shape %v", strides, kernel)
		}
		input := n.Inputs[0]
		var recorded int64
		if t, ok := g.TypeOf(input); ok {
			if t.ElemType != graph.Float {
				return AveragePoolReport{}, fail("input %q is %s, and the weight is float32", input, t.ElemType)
			}
			if t.Ranked {
				if len(t.Shape) != len(kernel)+2 {
					return AveragePoolReport{}, fail("input %q has rank %d, and kernel_shape %v needs %d", input, len(t.Shape), kernel, len(kernel)+2)
				}
				recorded = t.Shape[1].Value
			}
		} else if _, hinted := opts.InputChannels[input]; !hinted {
			return AveragePoolReport{}, fail("the graph does not record the type of %q and it is not float32 for certain; give the channels with a hint", input)
		}
		channels := recorded
		if hint, ok := opts.InputChannels[input]; ok {
			if recorded > 0 && recorded != hint {
				return AveragePoolReport{}, fail("the graph records %d channels for %q and the hint says %d", recorded, input, hint)
			}
			channels = hint
		}
		if channels <= 0 {
			return AveragePoolReport{}, fail("the channels of %q (dimension 1) are not known: the graph does not record them and no hint was given", input)
		}
		plans = append(plans, plan{n, channels, kernel, strides})
	}
	if len(plans) == 0 {
		report.NodesAfter = len(g.Nodes)
		return report, nil
	}

	namer := graph.NewNamer(g)
	weights := map[string]string{}
	for _, p := range plans {
		n := p.node
		key := fmt.Sprint(p.channels, p.kernel)
		weight, ok := weights[key]
		if !ok {
			weight = namer.Fresh(n.Name + "/mean_weight")
			weights[key] = weight
			values := make([]float32, p.channels*product(p.kernel))
			for i := range values {
				values[i] = 1 / float32(product(p.kernel))
			}
			dims := append([]int64{p.channels, 1}, p.kernel...)
			g.Initializers = append(g.Initializers, graph.Float32Tensor(weight, dims, values))
		}
		name := n.Name
		if name != "" {
			name = namer.Fresh(name + "/depthwise_conv")
		}
		rec := AveragePoolRewrite{
			Node: n.Name, Conv: name, Input: n.Inputs[0], Output: n.Outputs[0],
			Channels: p.channels, Kernel: p.kernel, Strides: p.strides, Weight: weight,
		}
		n.OpType = "Conv"
		n.Name = name
		n.Inputs = []string{rec.Input, weight}
		n.Attributes = []*graph.Attribute{
			graph.IntAttribute("group", p.channels),
			graph.IntsAttribute("kernel_shape", p.kernel),
			graph.IntsAttribute("pads", make([]int64, 2*len(p.kernel))),
			graph.IntsAttribute("strides", p.strides),
		}
		report.Rewritten = append(report.Rewritten, rec)
	}
	report.NodesAfter = len(g.Nodes)
	if err := g.CheckTopological(); err != nil {
		return AveragePoolReport{}, fmt.Errorf("rewrite: the rewritten graph is out of order, which is a defect of the rewrite: %w", err)
	}
	return report, nil
}
