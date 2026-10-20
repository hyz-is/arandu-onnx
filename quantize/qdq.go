// Package quantize exports a float ONNX model as an explicitly quantized
// (QDQ) INT8 model in the form TensorRT runs, from a calibration table, in
// pure Go.
//
// TensorRT 10 runs INT8 two ways. Implicit quantization (a float model plus
// a calibration table) is deprecated and gone in TensorRT 11. Explicit
// quantization reads QuantizeLinear and DequantizeLinear nodes from the
// graph, and accepts them only when they are symmetric INT8: an int8 zero
// point of 0, a positive scale, per-tensor for activations and per-channel
// for the weights of convolutions and matrix products, and no
// DequantizeLinear on a bias. QDQ writes exactly that, and CheckTensorRT
// checks any model, this package's or another's, against the same rules.
//
// The scales come from a calibration.Table (Runtime.Calibrate or the
// arandu-onnx-calibrate command): an activation of range r gets the scale
// r / 127. The weights are quantized here, offline, into int8 initializers
// read through a DequantizeLinear; their float initializers are removed when
// nothing else reads them. Biases stay float32: TensorRT fuses them itself
// and refuses a quantized one.
package quantize

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/hyz-is/arandu-onnx/calibration"
	"github.com/hyz-is/arandu-onnx/graph"
)

// ErrRefused is wrapped by every error QDQ returns for a model or options it
// will not quantize. The model is then left unchanged.
var ErrRefused = errors.New("quantize: refused")

// Mode is which operators QDQ quantizes.
type Mode string

// The modes QDQ takes.
const (
	// ModeConv quantizes Conv and ConvTranspose only, the default. It is
	// where INT8 pays on a convolutional network, and it leaves matrix
	// products (attention, heads) in float.
	ModeConv Mode = "conv"
	// ModeConvMatMul also quantizes MatMul and Gemm whose second input is a
	// constant 2-D weight. A MatMul of two activations (attention) is never
	// quantized.
	ModeConvMatMul Mode = "conv-matmul"
)

// Options says what QDQ quantizes.
type Options struct {
	// Mode is which operators are quantized. Empty means ModeConv.
	Mode Mode
	// ExcludeNodes is nodes left in float, by exact name.
	ExcludeNodes []string
	// ExcludePrefixes is nodes left in float, by name prefix, such as a
	// detector head "/model/model.22/".
	ExcludePrefixes []string
	// ExcludeOpTypes is operators left in float, such as ConvTranspose.
	ExcludeOpTypes []string
}

// MinimumOpset is the default-domain opset QDQ writes at least: per-channel
// DequantizeLinear (its axis attribute) is opset 13.
const MinimumOpset = 13

// Report is what QDQ did.
type Report struct {
	// Mode is the mode applied.
	Mode Mode `json:"mode"`
	// OpsetBefore and OpsetAfter are the default-domain opset; they differ
	// when QDQ raised it to MinimumOpset.
	OpsetBefore int64 `json:"opset_before"`
	OpsetAfter  int64 `json:"opset_after"`
	// NodesBefore and NodesAfter are the graph's node counts.
	NodesBefore int `json:"nodes_before"`
	NodesAfter  int `json:"nodes_after"`
	// Quantized is every node that now reads quantized inputs, in graph
	// order.
	Quantized []QuantizedNode `json:"quantized"`
	// Skipped is every node of a quantized operator left in float, and why.
	Skipped []SkippedNode `json:"skipped"`
	// Activations is every activation quantized, in name order.
	Activations []Activation `json:"activations"`
	// Weights is every weight quantized, in name order.
	Weights []Weight `json:"weights"`
	// RemovedInitializers is the float weights nothing reads any more:
	// initializers, and the outputs of Constant nodes removed with them.
	RemovedInitializers []string `json:"removed_initializers"`
	// DetachedBiases is the biases moved out of the reach of ONNX Runtime's
	// WeightBiasQuantization, which would make them int32 DequantizeLinear
	// nodes TensorRT does not run: a Conv or ConvTranspose bias now added
	// after the node, a Gemm or MatMul bias reshaped to [1, N].
	DetachedBiases []string `json:"detached_biases"`
	// ScalesSHA256 is a digest of every scale written: the SHA-256 of the
	// sorted lines "a <tensor> <scale bits>" and "w <weight> <axis> <scale
	// bits>...", the bits in hexadecimal. Two exports with the same digest
	// quantize with the same numbers.
	ScalesSHA256 string `json:"scales_sha256"`
}

// QuantizedNode is one node QDQ quantized.
type QuantizedNode struct {
	Node       string `json:"node"`
	OpType     string `json:"op_type"`
	Activation string `json:"activation"`
	Weight     string `json:"weight"`
}

// SkippedNode is one node of a quantized operator left in float.
type SkippedNode struct {
	Node   string `json:"node"`
	OpType string `json:"op_type"`
	Reason string `json:"reason"`
}

// Activation is one activation and its per-tensor scale.
type Activation struct {
	Tensor string  `json:"tensor"`
	Range  float32 `json:"range"`
	Scale  float32 `json:"scale"`
}

// Weight is one weight and its per-channel quantization.
type Weight struct {
	// Initializer is the float weight; Quantized is the int8 initializer
	// that replaced it.
	Initializer string `json:"initializer"`
	Quantized   string `json:"quantized"`
	// Axis is the channel axis, and Channels its size.
	Axis     int `json:"axis"`
	Channels int `json:"channels"`
	// MaxError is the largest |w - dequantized w| over the weight, at most
	// half the largest scale.
	MaxError float32 `json:"max_error"`
}

// QDQ quantizes the model in place, from the table's activation ranges, into
// the explicit INT8 form TensorRT runs.
//
// For each node of the mode's operators that no exclusion names, whose
// activation input the table gives a range, and whose weight is a constant
// float tensor (an initializer that is not also a graph input, or the value
// of a Constant node, as some exporters write their weights):
//
//   - the activation is read through QuantizeLinear and DequantizeLinear,
//     per tensor, scale range / 127, int8 zero point 0; one pair per tensor,
//     shared by every quantized node that reads it, while the nodes left in
//     float keep reading the float tensor;
//   - the weight becomes an int8 initializer, per channel along the output
//     channels (axis 0 for Conv, 1 for ConvTranspose and MatMul, 0 or 1 for
//     Gemm by transB), scale max|w| / 127, values rounded half to even into
//     [-127, 127], read through a DequantizeLinear with int8 zero points 0;
//   - the bias is left as it is, float32, with nothing in front of it.
//
// A model whose default-domain opset is below 13 is raised to 13 when every
// operator it uses means the same there (from opset 10 to 12, with none of
// the operators whose definition changed before 13); otherwise it is
// refused. A model already quantized, a table that names no tensor of the
// graph, weights kept in external data, or options that leave nothing to
// quantize are refused with ErrRefused, and the model is left unchanged.
// Nodes inside subgraphs are not quantized.
func QDQ(m *graph.Model, table calibration.Table, opts Options) (Report, error) {
	if opts.Mode == "" {
		opts.Mode = ModeConv
	}
	if opts.Mode != ModeConv && opts.Mode != ModeConvMatMul {
		return Report{}, fmt.Errorf("%w: mode %q is not %q or %q", ErrRefused, opts.Mode, ModeConv, ModeConvMatMul)
	}
	if m == nil || m.Graph == nil {
		return Report{}, fmt.Errorf("%w: the model has no graph", ErrRefused)
	}
	if err := table.Validate(); err != nil {
		return Report{}, fmt.Errorf("%w: %v", ErrRefused, err)
	}
	g := m.Graph
	for _, n := range g.Nodes {
		if quantizedOps[n.OpType] && n.IsStandard() {
			return Report{}, fmt.Errorf("%w: the model is already quantized (%s %q); QDQ starts from the float model", ErrRefused, n.OpType, n.Name)
		}
	}
	named := false
	for _, name := range table.Names() {
		if g.Producer(name) != nil || g.IsInput(name) {
			named = true
			break
		}
	}
	if !named {
		return Report{}, fmt.Errorf("%w: none of the tensors of the table is a tensor of the graph: the table belongs to another model, or to another export of this one", ErrRefused)
	}
	opsetBefore, opsetAfter, err := opsetPlan(m)
	if err != nil {
		return Report{}, err
	}

	report := Report{Mode: opts.Mode, OpsetBefore: opsetBefore, OpsetAfter: opsetAfter, NodesBefore: len(g.Nodes)}
	plans, skipped, err := planNodes(g, table, opts)
	if err != nil {
		return Report{}, err
	}
	report.Skipped = skipped
	if len(plans) == 0 {
		return Report{}, fmt.Errorf("%w: no node is left to quantize (%d skipped; see the reasons in the report of a narrower run)", ErrRefused, len(skipped))
	}

	// Every check has passed: from here on the model is changed.
	namer := graph.NewNamer(g)
	before := map[*graph.Node][]*graph.Node{}
	activations := map[string]string{} // float tensor -> dequantized tensor
	weights := map[string]string{}     // weight@axis -> dequantized tensor
	var digest []string

	for _, p := range plans {
		dq, ok := activations[p.activation]
		if !ok {
			r := table[p.activation]
			scale := r / 127
			scaleName := namer.Fresh(p.activation + "_scale")
			zeroName := namer.Fresh(p.activation + "_zero_point")
			quantized := namer.Fresh(p.activation + "_quantized")
			dq = namer.Fresh(p.activation + "_dequantized")
			g.Initializers = append(g.Initializers,
				graph.Float32Tensor(scaleName, nil, []float32{scale}),
				graph.Int8Tensor(zeroName, nil, []int8{0}))
			before[p.node] = append(before[p.node],
				&graph.Node{
					Name:    namer.Fresh(p.activation + "_QuantizeLinear"),
					OpType:  "QuantizeLinear",
					Inputs:  []string{p.activation, scaleName, zeroName},
					Outputs: []string{quantized},
				},
				&graph.Node{
					Name:    namer.Fresh(p.activation + "_DequantizeLinear"),
					OpType:  "DequantizeLinear",
					Inputs:  []string{quantized, scaleName, zeroName},
					Outputs: []string{dq},
				})
			activations[p.activation] = dq
			report.Activations = append(report.Activations, Activation{Tensor: p.activation, Range: r, Scale: scale})
			digest = append(digest, "a "+p.activation+" "+bits(scale))
		}

		key := p.weight.Name + "@" + strconv.Itoa(p.axis)
		wdq, ok := weights[key]
		if !ok {
			q, scales, maxErr := quantizeWeight(p.values, p.weight.Dims, p.axis)
			qName := namer.Fresh(p.weight.Name + "_quantized")
			scaleName := namer.Fresh(p.weight.Name + "_scale")
			zeroName := namer.Fresh(p.weight.Name + "_zero_point")
			wdq = namer.Fresh(p.weight.Name + "_dequantized")
			channels := int64(len(scales))
			g.Initializers = append(g.Initializers,
				graph.Int8Tensor(qName, p.weight.Dims, q),
				graph.Float32Tensor(scaleName, []int64{channels}, scales),
				graph.Int8Tensor(zeroName, []int64{channels}, make([]int8, channels)))
			before[p.node] = append(before[p.node], &graph.Node{
				Name:       namer.Fresh(p.weight.Name + "_DequantizeLinear"),
				OpType:     "DequantizeLinear",
				Inputs:     []string{qName, scaleName, zeroName},
				Outputs:    []string{wdq},
				Attributes: []*graph.Attribute{graph.IntAttribute("axis", int64(p.axis))},
			})
			weights[key] = wdq
			report.Weights = append(report.Weights, Weight{
				Initializer: p.weight.Name, Quantized: qName, Axis: p.axis, Channels: len(scales), MaxError: maxErr,
			})
			line := "w " + qName + " " + strconv.Itoa(p.axis)
			for _, s := range scales {
				line += " " + bits(s)
			}
			digest = append(digest, line)
		}

		p.node.Inputs[p.activationIndex] = dq
		p.node.Inputs[p.weightIndex] = wdq
		report.Quantized = append(report.Quantized, QuantizedNode{
			Node: p.node.Name, OpType: p.node.OpType, Activation: p.activation, Weight: p.weight.Name,
		})
	}

	nodes := make([]*graph.Node, 0, len(g.Nodes)+2*len(activations)+len(weights))
	for _, n := range g.Nodes {
		nodes = append(nodes, before[n]...)
		nodes = append(nodes, n)
	}
	g.Nodes = nodes

	report.DetachedBiases = detachBiases(g, namer)

	if !g.HasSubgraphs() {
		seen := map[string]bool{}
		for _, p := range plans {
			name := p.weight.Name
			if seen[name] || g.Used(name) || g.IsInput(name) {
				continue
			}
			seen[name] = true
			if g.Initializer(name) != nil {
				g.RemoveInitializer(name)
			} else if c := g.Producer(name); c != nil && c.OpType == "Constant" {
				kept := g.Nodes[:0]
				for _, n := range g.Nodes {
					if n != c {
						kept = append(kept, n)
					}
				}
				g.Nodes = kept
			}
			report.RemovedInitializers = append(report.RemovedInitializers, name)
		}
		sort.Strings(report.RemovedInitializers)
	}
	if opsetAfter != opsetBefore {
		m.SetOpset(graph.DefaultDomain, opsetAfter)
		if m.IRVersion < 7 {
			m.IRVersion = 7
		}
	}
	if err := g.CheckTopological(); err != nil {
		return Report{}, fmt.Errorf("quantize: the quantized graph is out of order, which is a defect of QDQ: %w", err)
	}

	sort.Slice(report.Activations, func(i, j int) bool { return report.Activations[i].Tensor < report.Activations[j].Tensor })
	sort.Slice(report.Weights, func(i, j int) bool { return report.Weights[i].Initializer < report.Weights[j].Initializer })
	sort.Strings(digest)
	sum := sha256.Sum256([]byte(strings.Join(digest, "\n") + "\n"))
	report.ScalesSHA256 = hex.EncodeToString(sum[:])
	report.NodesAfter = len(g.Nodes)
	return report, nil
}

// quantizedOps are the operators of a model already quantized, in either
// form.
var quantizedOps = map[string]bool{
	"QuantizeLinear": true, "DequantizeLinear": true, "DynamicQuantizeLinear": true,
	"QLinearConv": true, "QLinearMatMul": true, "ConvInteger": true, "MatMulInteger": true,
}

// plan is one node QDQ will quantize.
type plan struct {
	node            *graph.Node
	activation      string
	activationIndex int
	weight          *graph.Tensor
	weightIndex     int
	axis            int
	values          []float32
}

// planNodes decides, node by node, what is quantized and what is skipped.
func planNodes(g *graph.Graph, table calibration.Table, opts Options) ([]plan, []SkippedNode, error) {
	ops := map[string]bool{"Conv": true, "ConvTranspose": true}
	if opts.Mode == ModeConvMatMul {
		ops["MatMul"] = true
		ops["Gemm"] = true
	}
	var plans []plan
	var skipped []SkippedNode
	skip := func(n *graph.Node, reason string) {
		skipped = append(skipped, SkippedNode{Node: n.Name, OpType: n.OpType, Reason: reason})
	}
	for _, n := range g.Nodes {
		if !ops[n.OpType] || !n.IsStandard() {
			continue
		}
		if reason := excluded(n, opts); reason != "" {
			skip(n, reason)
			continue
		}
		activation := n.Input(0)
		if g.Constant(activation) != nil {
			skip(n, "its first input is a constant, not an activation")
			continue
		}
		if _, ok := table[activation]; !ok {
			skip(n, fmt.Sprintf("the table gives no range for its input %q (excluded from calibration, or not float32)", activation))
			continue
		}
		weight := g.Constant(n.Input(1))
		if weight == nil {
			if g.Initializer(n.Input(1)) != nil && g.IsInput(n.Input(1)) {
				skip(n, fmt.Sprintf("its weight %q is an initializer listed as a graph input, which a caller may override, so it is not a constant: rewrite.InitializersOutOfInputs (arandu-onnx-rewrite -initializers-out-of-inputs) makes it one", n.Input(1)))
				continue
			}
			skip(n, fmt.Sprintf("its second input %q is not a constant weight (a product of two activations stays in float)", n.Input(1)))
			continue
		}
		if weight.External {
			return nil, nil, fmt.Errorf("%w: the weight %q of %s %q is in external data, which this package does not read", ErrRefused, weight.Name, n.OpType, n.Name)
		}
		if weight.DataType != graph.Float {
			skip(n, fmt.Sprintf("its weight %q is %s, not float32", weight.Name, weight.DataType))
			continue
		}
		values, err := weight.Float32s()
		if err != nil {
			return nil, nil, fmt.Errorf("%w: %v", ErrRefused, err)
		}
		axis := 0
		switch n.OpType {
		case "Conv":
			if len(weight.Dims) < 3 {
				skip(n, fmt.Sprintf("its weight has shape %v, and a Conv weight is [M, C/group, k...]", weight.Dims))
				continue
			}
		case "ConvTranspose":
			if len(weight.Dims) < 3 {
				skip(n, fmt.Sprintf("its weight has shape %v, and a ConvTranspose weight is [C, M/group, k...]", weight.Dims))
				continue
			}
			axis = 1
		case "MatMul":
			if len(weight.Dims) != 2 {
				skip(n, fmt.Sprintf("its weight has shape %v, and TensorRT quantizes a 2-D MatMul weight per channel", weight.Dims))
				continue
			}
			axis = 1
		case "Gemm":
			if len(weight.Dims) != 2 {
				skip(n, fmt.Sprintf("its weight has shape %v", weight.Dims))
				continue
			}
			if n.Int("transB", 0) == 0 {
				axis = 1
			}
		}
		plans = append(plans, plan{
			node: n, activation: activation, activationIndex: 0,
			weight: weight, weightIndex: 1, axis: axis, values: values,
		})
	}
	return plans, skipped, nil
}

// excluded says why the options leave a node in float, or "".
func excluded(n *graph.Node, opts Options) string {
	for _, name := range opts.ExcludeNodes {
		if n.Name == name {
			return "excluded by name"
		}
	}
	for _, prefix := range opts.ExcludePrefixes {
		if prefix != "" && strings.HasPrefix(n.Name, prefix) {
			return "excluded by the prefix " + strconv.Quote(prefix)
		}
	}
	for _, op := range opts.ExcludeOpTypes {
		if n.OpType == op {
			return "excluded by operator"
		}
	}
	return ""
}

// quantizeWeight quantizes a weight symmetrically per channel along an axis.
// It returns the int8 values, one scale per channel, and the largest error.
func quantizeWeight(values []float32, dims []int64, axis int) ([]int8, []float32, float32) {
	channels := int(dims[axis])
	inner := 1
	for _, d := range dims[axis+1:] {
		inner *= int(d)
	}
	channelOf := func(i int) int { return (i / inner) % channels }

	peak := make([]float64, channels)
	for i, v := range values {
		c := channelOf(i)
		peak[c] = math.Max(peak[c], math.Abs(float64(v)))
	}
	scales := make([]float32, channels)
	for c, p := range peak {
		s := float32(p / 127)
		// A channel of zeros, or of values too small for a normal float32
		// scale, quantizes to zeros with scale 1: TensorRT takes only
		// positive scales.
		if p == 0 || s < math.SmallestNonzeroFloat32*(1<<23) {
			s = 1
		}
		scales[c] = s
	}
	q := make([]int8, len(values))
	var maxErr float64
	for i, v := range values {
		s := scales[channelOf(i)]
		r := math.RoundToEven(float64(v) / float64(s))
		r = math.Max(-127, math.Min(127, r))
		q[i] = int8(r)
		maxErr = math.Max(maxErr, math.Abs(float64(v)-r*float64(s)))
	}
	return q, scales, float32(maxErr)
}

// bits is a float32's IEEE-754 bits in hexadecimal.
func bits(v float32) string { return fmt.Sprintf("%08x", math.Float32bits(v)) }
