package quantize

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/hyz-is/arandu-onnx/graph"
)

// QNNType is the integer type the activations of a QNN export are written in.
type QNNType string

// The activation types QNN takes.
const (
	// QNNUint8 is 8-bit unsigned activations, the default: the form a
	// Qualcomm HTP runs fastest.
	QNNUint8 QNNType = "uint8"
	// QNNUint16 is 16-bit unsigned activations. The quantize and dequantize
	// nodes are written in the com.microsoft domain, which takes uint16 at
	// any opset; the weights stay uint8.
	QNNUint16 QNNType = "uint16"
)

// Range is the smallest and the largest value a float tensor took over the
// calibration inputs.
type Range struct {
	Min float32 `json:"min"`
	Max float32 `json:"max"`
}

// QParams is the quantization of one tensor: value = (q - ZeroPoint) * Scale.
type QParams struct {
	// Type is the integer type; empty means the export's activation type.
	Type QNNType `json:"type,omitempty"`
	// Scale is positive and ZeroPoint lies in the type's range.
	Scale     float32 `json:"scale"`
	ZeroPoint int32   `json:"zero_point"`
}

// QNNOptions says how QNN quantizes.
type QNNOptions struct {
	// Activations is the activation type. Empty means QNNUint8.
	Activations QNNType `json:"activations,omitempty"`
	// Ranges is the measured minimum and maximum of each float tensor, by
	// name. A tensor a quantized node reads or writes needs one, unless
	// Fixed names it or it is derived (the output of a Sigmoid, Softmax or
	// Tanh, or of an operator that moves data without computing, which take
	// their scale from the operator and from the input).
	Ranges map[string]Range `json:"ranges"`
	// Fixed pins the quantization of the named tensors, overriding the
	// range and the derivation. A pinned graph input listed in
	// IntegerInputs is what makes the model take raw integers.
	Fixed map[string]QParams `json:"fixed,omitempty"`
	// IntegerInputs are graph inputs the model reads as integers: the input
	// becomes uint8, has no QuantizeLinear, and its DequantizeLinear reads it
	// directly. Each needs a Fixed entry of type uint8 with zero point 0 or
	// any valid zero point, such as scale 1/255 for an image read as bytes.
	IntegerInputs []string `json:"integer_inputs,omitempty"`
	// WeightScales gives a weight (by initializer name) one scale for every
	// channel, in place of 2*max|w|/255 per channel, with zero point 128. A
	// weight that is a power of two times a small integer, such as the 0.25 of
	// a mean, is then exact.
	WeightScales map[string]float32 `json:"weight_scales,omitempty"`
	// ExcludeNodes, ExcludePrefixes and ExcludeOpTypes leave nodes in float,
	// as in Options.
	ExcludeNodes    []string `json:"exclude_nodes,omitempty"`
	ExcludePrefixes []string `json:"exclude_prefixes,omitempty"`
	ExcludeOpTypes  []string `json:"exclude_op_types,omitempty"`
}

// MinimumRealRange is the narrowest range QNN gives an activation: a tensor
// whose values are nearly constant is widened to this much, as ONNX Runtime's
// quantizer does.
const MinimumRealRange float32 = 0.0001

// QNNTensor is one tensor QNN quantized and how its parameters were found.
type QNNTensor struct {
	Tensor    string  `json:"tensor"`
	Type      QNNType `json:"type"`
	Scale     float32 `json:"scale"`
	ZeroPoint int32   `json:"zero_point"`
	// Source is "range" (computed from Ranges), "fixed" (Options.Fixed),
	// "operator" (the fixed scale of Sigmoid, Softmax or Tanh) or "input"
	// (copied from the input of an operator that moves data).
	Source string `json:"source"`
}

// QNNWeight is one weight and its per-channel quantization.
type QNNWeight struct {
	Initializer string `json:"initializer"`
	Quantized   string `json:"quantized"`
	Axis        int    `json:"axis"`
	Channels    int    `json:"channels"`
	// MaxError is the largest |w - dequantized w|.
	MaxError float32 `json:"max_error"`
}

// QNNBias is one bias and its int32 quantization, scale = input scale * weight
// scale per channel.
type QNNBias struct {
	Initializer string `json:"initializer"`
	Quantized   string `json:"quantized"`
}

// QNNReport is what QNN did.
type QNNReport struct {
	Activations QNNType `json:"activations"`
	// OpsetBefore and OpsetAfter are the default-domain opset.
	OpsetBefore int64 `json:"opset_before"`
	OpsetAfter  int64 `json:"opset_after"`
	NodesBefore int   `json:"nodes_before"`
	NodesAfter  int   `json:"nodes_after"`
	// Quantized is the nodes that read and write quantized tensors, in graph
	// order, and Skipped those left in float and why.
	Quantized []string      `json:"quantized"`
	Skipped   []SkippedNode `json:"skipped"`
	// Tensors, Weights and Biases are what was quantized, in name order.
	Tensors []QNNTensor `json:"tensors"`
	Weights []QNNWeight `json:"weights"`
	Biases  []QNNBias   `json:"biases"`
	// RemovedInitializers is the float weights and biases nothing reads any
	// more.
	RemovedInitializers []string `json:"removed_initializers"`
	// ScalesSHA256 is a digest of every scale and zero point written, so two
	// exports can be compared: the SHA-256 of the sorted lines "a <tensor>
	// <type> <scale bits> <zero point>" and "w <weight> <axis> <scale bits>...".
	ScalesSHA256 string `json:"scales_sha256"`
}

// qnnActivationOps are the operators QNN reads and writes quantized, with the
// inputs that are activations (the others, such as the shape of a Reshape or
// the scales of a Resize, are constants of another type and stay as they are).
var qnnActivationOps = map[string]int{
	// number of leading inputs that are activations; -1 means all of them
	"Conv": 1, "ConvTranspose": 1,
	"Add": -1, "Sub": -1, "Mul": -1, "Div": -1, "Concat": -1,
	"Sigmoid": 1, "Tanh": 1, "Softmax": 1, "Relu": 1, "Clip": 1,
	"Slice": 1, "Split": 1, "Resize": 1, "MaxPool": 1, "AveragePool": 1,
	"Reshape": 1, "Transpose": 1, "Squeeze": 1, "Unsqueeze": 1, "Flatten": 1,
}

// qnnDirectOps move or select data without computing a new value, so their
// output is quantized exactly as their input: the integers pass through.
var qnnDirectOps = map[string]bool{
	"Slice": true, "Split": true, "Resize": true, "MaxPool": true,
	"Reshape": true, "Transpose": true, "Squeeze": true, "Unsqueeze": true, "Flatten": true,
}

// qnnConvOps carry a weight (and a bias) quantized per output channel.
var qnnConvOps = map[string]int{"Conv": 0, "ConvTranspose": 1}

// QNN quantizes the model in place into the QDQ form the Qualcomm QNN
// execution provider runs on the Hexagon HTP: asymmetric unsigned activations
// per tensor, per-channel unsigned weights and int32 biases.
//
// The form is the one ONNX Runtime's QNN quantization writes, and differs from
// QDQ (the symmetric int8 form TensorRT runs) in these ways:
//
//   - an activation of range [min, max] gets the scale (max - min) / (qmax - qmin)
//     with the range widened to include 0 and to MinimumRealRange, and the zero
//     point round(qmin - min / scale), clamped (uint8: 0..255, uint16: 0..65535);
//   - the output of Sigmoid and Softmax has the scale 1/256 and zero point 0
//     (1/65536 for uint16), and the output of Tanh 1/128 and 128 (1/32768 and
//     32768), which are what the HTP requires of them;
//   - the output of an operator that moves data (Slice, Split, Resize,
//     MaxPool, Reshape, Transpose, Squeeze, Unsqueeze, Flatten) is quantized as
//     its input;
//   - a Conv or ConvTranspose weight is uint8 per output channel (axis 0, or 1
//     for ConvTranspose) with zero point 128 and scale 2*max|w|/255, and its
//     bias is int32, per channel, with scale input scale * weight scale and zero
//     point 0, read by a DequantizeLinear with axis 0;
//   - a graph input listed in IntegerInputs is read as uint8 with no
//     QuantizeLinear in front of its DequantizeLinear.
//
// Each tensor gets one QuantizeLinear and one DequantizeLinear. The producer's
// output is renamed "<tensor>_QuantizeLinear_Input" and the DequantizeLinear
// writes the tensor under its own name, so every consumer and every graph
// output keeps reading and exposing the original name, as float.
//
// A node is quantized only when every activation it reads or writes has a
// quantization (from Ranges, Fixed or its operator); otherwise it is left in
// float and reported. A constant operand of Add, Sub, Mul, Div or Concat
// leaves that node in float. Operators outside qnnActivationOps (MatMul, Gemm
// and others) stay in float. A bias whose int32 value would overflow refuses
// the export, naming the input whose range has to widen. A model already
// quantized, one with subgraphs, or options that leave nothing to quantize are
// refused with ErrRefused and the model is left unchanged.
func QNN(m *graph.Model, opts QNNOptions) (QNNReport, error) {
	if m == nil || m.Graph == nil {
		return QNNReport{}, fmt.Errorf("%w: the model has no graph", ErrRefused)
	}
	act := opts.Activations
	if act == "" {
		act = QNNUint8
	}
	if act != QNNUint8 && act != QNNUint16 {
		return QNNReport{}, fmt.Errorf("%w: activations %q have to be %q or %q", ErrRefused, act, QNNUint8, QNNUint16)
	}
	g := m.Graph
	if g.HasSubgraphs() {
		return QNNReport{}, fmt.Errorf("%w: the model has subgraphs, which are not quantized", ErrRefused)
	}
	for _, n := range g.Nodes {
		if quantizedOps[n.OpType] && n.IsStandard() {
			return QNNReport{}, fmt.Errorf("%w: the model is already quantized (%s %q); QNN starts from the float model", ErrRefused, n.OpType, n.Name)
		}
	}
	for name, r := range opts.Ranges {
		if !finite(r.Min) || !finite(r.Max) || r.Min > r.Max {
			return QNNReport{}, fmt.Errorf("%w: the range of %q is [%g, %g]", ErrRefused, name, r.Min, r.Max)
		}
	}
	for name, p := range opts.Fixed {
		if err := validateQParams(name, p, act); err != nil {
			return QNNReport{}, err
		}
	}
	integer := map[string]bool{}
	for _, name := range opts.IntegerInputs {
		p, ok := opts.Fixed[name]
		if !g.IsInput(name) || g.Initializer(name) != nil {
			return QNNReport{}, fmt.Errorf("%w: IntegerInputs names %q, which is not a graph input", ErrRefused, name)
		}
		if !ok || p.typeOr(act) != QNNUint8 {
			return QNNReport{}, fmt.Errorf("%w: the integer input %q needs a Fixed entry of type %q", ErrRefused, name, QNNUint8)
		}
		integer[name] = true
	}
	opsetBefore, opsetAfter, err := opsetPlan(m)
	if err != nil {
		return QNNReport{}, err
	}

	report := QNNReport{Activations: act, OpsetBefore: opsetBefore, OpsetAfter: opsetAfter, NodesBefore: len(g.Nodes)}
	pl := planQNN(g, opts, act)
	report.Skipped = pl.skipped
	if len(pl.nodes) == 0 {
		return QNNReport{}, fmt.Errorf("%w: no node is left to quantize (%d skipped)", ErrRefused, len(pl.skipped))
	}

	// Parameters of every tensor, in graph order so a derived tensor finds
	// its input done.
	params := map[string]QParams{}
	source := map[string]string{}
	set := func(name string, p QParams, why string) { params[name] = p; source[name] = why }
	for _, name := range pl.order {
		if p, ok := opts.Fixed[name]; ok {
			p.Type = p.typeOr(act)
			set(name, p, "fixed")
			continue
		}
		prod := g.Producer(name)
		if prod != nil && prod.IsStandard() {
			switch prod.OpType {
			case "Sigmoid", "Softmax":
				set(name, fixedOutput(prod.OpType, act), "operator")
				continue
			case "Tanh":
				set(name, fixedOutput(prod.OpType, act), "operator")
				continue
			}
			if qnnDirectOps[prod.OpType] && pl.nodes[prod] {
				if in, ok := params[prod.Input(0)]; ok {
					set(name, in, "input")
					continue
				}
			}
		}
		r, ok := opts.Ranges[name]
		if !ok {
			return QNNReport{}, fmt.Errorf("%w: no range for %q, which a quantized node needs (a defect of the plan)", ErrRefused, name)
		}
		p, err := rangeParams(name, r, act)
		if err != nil {
			return QNNReport{}, err
		}
		set(name, p, "range")
	}

	// Weights and biases first, so a refusal leaves the model unchanged.
	type convWeights struct {
		node         *graph.Node
		weight       *graph.Tensor
		values, bias []float32
		biasTensor   *graph.Tensor
		axis         int
		q            []uint8
		scales       []float32
		maxErr       float32
		biasQ        []int32
		biasScales   []float32
	}
	var convs []*convWeights
	for _, n := range g.Nodes {
		axis, isConv := qnnConvOps[n.OpType]
		if !isConv || !pl.nodes[n] {
			continue
		}
		w := g.Constant(n.Input(1))
		values, err := w.Float32s()
		if err != nil {
			return QNNReport{}, fmt.Errorf("%w: %v", ErrRefused, err)
		}
		c := &convWeights{node: n, weight: w, values: values, axis: axis}
		c.q, c.scales, c.maxErr = quantizeWeightUint8(values, w.Dims, axis, opts.WeightScales[w.Name])
		if bias := n.Input(2); bias != "" {
			bt := g.Constant(bias)
			bv, err := bt.Float32s()
			if err != nil {
				return QNNReport{}, fmt.Errorf("%w: %v", ErrRefused, err)
			}
			c.biasTensor, c.bias = bt, bv
			sx := float64(params[n.Input(0)].Scale)
			if len(c.scales) != len(bv) {
				return QNNReport{}, fmt.Errorf("%w: the bias %q of %s %q has %d values and its weight %d output channels", ErrRefused, bt.Name, n.OpType, n.Name, len(bv), len(c.scales))
			}
			c.biasScales = make([]float32, len(bv))
			c.biasQ = make([]int32, len(bv))
			for i, b := range bv {
				// the bias channel i belongs to the weight's output channel i
				sw := float64(c.scales[i])
				scale := float32(sx * sw)
				c.biasScales[i] = scale
				q := math.RoundToEven(float64(b / scale))
				if q > math.MaxInt32 || q < math.MinInt32 {
					return QNNReport{}, fmt.Errorf("%w: the bias %q of %s %q overflows int32 at the scale %g (input %q has scale %g): widen the range of %q", ErrRefused, bt.Name, n.OpType, n.Name, scale, n.Input(0), sx, n.Input(0))
				}
				c.biasQ[i] = int32(q)
			}
		}
		convs = append(convs, c)
	}

	// Every check has passed: from here on the model is changed.
	namer := graph.NewNamer(g)
	var digest []string
	before := map[*graph.Node][]*graph.Node{}
	var head []*graph.Node
	after := map[*graph.Node][]*graph.Node{}
	deq := map[string]string{} // graph input -> its dequantized tensor

	for _, name := range pl.order {
		p := params[name]
		scaleName := namer.Fresh(name + "_scale")
		zeroName := namer.Fresh(name + "_zero_point")
		g.Initializers = append(g.Initializers, graph.Float32Tensor(scaleName, nil, []float32{p.Scale}), zeroTensor(zeroName, p.Type, p.ZeroPoint))
		domain := ""
		if p.Type == QNNUint16 {
			domain = "com.microsoft"
		}
		quantized := namer.Fresh(name + "_quantized")
		var nodes []*graph.Node
		producer := g.Producer(name)
		in := name
		out := name
		switch {
		case integer[name]:
			quantized = name // the input itself is the integer tensor
			out = namer.Fresh(name + "_DequantizeLinear_Output")
			deq[name] = out
		case producer == nil: // a graph input
			out = namer.Fresh(name + "_DequantizeLinear_Output")
			deq[name] = out
		default:
			in = namer.Fresh(name + "_QuantizeLinear_Input")
			for i, o := range producer.Outputs {
				if o == name {
					producer.Outputs[i] = in
				}
			}
		}
		if !integer[name] {
			nodes = append(nodes, &graph.Node{Name: namer.Fresh(name + "_QuantizeLinear"), OpType: "QuantizeLinear", Domain: domain,
				Inputs: []string{in, scaleName, zeroName}, Outputs: []string{quantized}})
		}
		nodes = append(nodes, &graph.Node{Name: namer.Fresh(name + "_DequantizeLinear"), OpType: "DequantizeLinear", Domain: domain,
			Inputs: []string{quantized, scaleName, zeroName}, Outputs: []string{out}})
		if producer == nil {
			head = append(head, nodes...)
		} else {
			after[producer] = append(after[producer], nodes...)
		}
		report.Tensors = append(report.Tensors, QNNTensor{Tensor: name, Type: p.Type, Scale: p.Scale, ZeroPoint: p.ZeroPoint, Source: source[name]})
		digest = append(digest, fmt.Sprintf("a %s %s %s %d", name, p.Type, bits(p.Scale), p.ZeroPoint))
	}
	// Consumers of a graph input read its dequantized tensor.
	for _, n := range g.Nodes {
		for i, in := range n.Inputs {
			if d, ok := deq[in]; ok {
				n.Inputs[i] = d
			}
		}
	}
	for name := range integer {
		for i, v := range g.Inputs {
			if v.Name != name {
				continue
			}
			t, _ := g.TypeOf(name)
			t.ElemType = graph.Uint8
			g.Inputs[i] = graph.NewValueInfo(name, t)
		}
	}

	for _, c := range convs {
		n := c.node
		w := c.weight
		qName := namer.Fresh(w.Name + "_quantized")
		sName := namer.Fresh(w.Name + "_scale")
		zName := namer.Fresh(w.Name + "_zero_point")
		wdq := namer.Fresh(w.Name + "_dequantized")
		channels := int64(len(c.scales))
		zeros := make([]uint8, channels)
		for i := range zeros {
			zeros[i] = 128
		}
		g.Initializers = append(g.Initializers,
			&graph.Tensor{Name: qName, Dims: append([]int64{}, w.Dims...), DataType: graph.Uint8, RawData: c.q},
			graph.Float32Tensor(sName, []int64{channels}, c.scales),
			&graph.Tensor{Name: zName, Dims: []int64{channels}, DataType: graph.Uint8, RawData: zeros})
		before[n] = append(before[n], &graph.Node{Name: namer.Fresh(w.Name + "_DequantizeLinear"), OpType: "DequantizeLinear",
			Inputs: []string{qName, sName, zName}, Outputs: []string{wdq}, Attributes: []*graph.Attribute{graph.IntAttribute("axis", int64(c.axis))}})
		n.Inputs[1] = wdq
		report.Weights = append(report.Weights, QNNWeight{Initializer: w.Name, Quantized: qName, Axis: c.axis, Channels: int(channels), MaxError: c.maxErr})
		line := "w " + w.Name + " " + strconv.Itoa(c.axis)
		for _, s := range c.scales {
			line += " " + bits(s)
		}
		digest = append(digest, line)

		if c.biasTensor != nil {
			bq := namer.Fresh(c.biasTensor.Name + "_quantized")
			bs := namer.Fresh(c.biasTensor.Name + "_quantized_scale")
			bz := namer.Fresh(c.biasTensor.Name + "_quantized_zero_point")
			bdq := namer.Fresh(c.biasTensor.Name + "_dequantized")
			raw := make([]byte, 0, 4*len(c.biasQ))
			for _, v := range c.biasQ {
				raw = binary.LittleEndian.AppendUint32(raw, uint32(v))
			}
			g.Initializers = append(g.Initializers,
				&graph.Tensor{Name: bq, Dims: []int64{int64(len(c.biasQ))}, DataType: graph.Int32, RawData: raw},
				graph.Float32Tensor(bs, []int64{int64(len(c.biasQ))}, c.biasScales),
				&graph.Tensor{Name: bz, Dims: []int64{int64(len(c.biasQ))}, DataType: graph.Int32, RawData: make([]byte, 4*len(c.biasQ))})
			before[n] = append(before[n], &graph.Node{Name: namer.Fresh(c.biasTensor.Name + "_DequantizeLinear"), OpType: "DequantizeLinear",
				Inputs: []string{bq, bs, bz}, Outputs: []string{bdq}, Attributes: []*graph.Attribute{graph.IntAttribute("axis", 0)}})
			n.Inputs[2] = bdq
			report.Biases = append(report.Biases, QNNBias{Initializer: c.biasTensor.Name, Quantized: bq})
		}
	}

	nodes := make([]*graph.Node, 0, len(g.Nodes)+2*len(report.Tensors)+2*len(convs))
	nodes = append(nodes, head...)
	for _, n := range g.Nodes {
		nodes = append(nodes, before[n]...)
		nodes = append(nodes, n)
		nodes = append(nodes, after[n]...)
	}
	g.Nodes = nodes

	// Remove the float weights and biases nothing reads any more.
	seen := map[string]bool{}
	for _, c := range convs {
		for _, t := range []*graph.Tensor{c.weight, c.biasTensor} {
			if t == nil || seen[t.Name] {
				continue
			}
			seen[t.Name] = true
			if g.Used(t.Name) || g.IsInput(t.Name) {
				continue
			}
			if g.Initializer(t.Name) != nil {
				g.RemoveInitializer(t.Name)
			} else if p := g.Producer(t.Name); p != nil && p.OpType == "Constant" {
				kept := g.Nodes[:0]
				for _, n := range g.Nodes {
					if n != p {
						kept = append(kept, n)
					}
				}
				g.Nodes = kept
			}
			report.RemovedInitializers = append(report.RemovedInitializers, t.Name)
		}
	}
	sort.Strings(report.RemovedInitializers)

	if opsetAfter != opsetBefore {
		m.SetOpset(graph.DefaultDomain, opsetAfter)
		if m.IRVersion < 7 {
			m.IRVersion = 7
		}
	}
	if act == QNNUint16 {
		if _, ok := m.Opset("com.microsoft"); !ok {
			m.SetOpset("com.microsoft", 1)
		}
	}
	if err := g.CheckTopological(); err != nil {
		return QNNReport{}, fmt.Errorf("quantize: the quantized graph is out of order, which is a defect of QNN: %w", err)
	}

	for _, n := range g.Nodes {
		if pl.nodes[n] {
			report.Quantized = append(report.Quantized, n.Name)
		}
	}
	sort.Slice(report.Tensors, func(i, j int) bool { return report.Tensors[i].Tensor < report.Tensors[j].Tensor })
	sort.Slice(report.Weights, func(i, j int) bool { return report.Weights[i].Initializer < report.Weights[j].Initializer })
	sort.Strings(digest)
	sum := sha256.Sum256([]byte(strings.Join(digest, "\n") + "\n"))
	report.ScalesSHA256 = hex.EncodeToString(sum[:])
	report.NodesAfter = len(g.Nodes)
	return report, nil
}

// qnnPlan is what QNN will quantize.
type qnnPlan struct {
	nodes   map[*graph.Node]bool
	order   []string // tensors that get a pair, in graph order
	skipped []SkippedNode
}

// planQNN decides which nodes are quantized and which tensors get a pair.
func planQNN(g *graph.Graph, opts QNNOptions, act QNNType) qnnPlan {
	pl := qnnPlan{nodes: map[*graph.Node]bool{}}
	known := map[string]bool{} // tensors that will have parameters
	for name := range opts.Fixed {
		known[name] = true
	}
	for name := range opts.Ranges {
		known[name] = true
	}
	skip := func(n *graph.Node, reason string) {
		pl.skipped = append(pl.skipped, SkippedNode{Node: n.Name, OpType: n.OpType, Reason: reason})
	}
	wanted := map[string]bool{}
	for _, n := range g.Nodes {
		count, ok := qnnActivationOps[n.OpType]
		if !ok || !n.IsStandard() {
			continue
		}
		if reason := excluded(n, Options{ExcludeNodes: opts.ExcludeNodes, ExcludePrefixes: opts.ExcludePrefixes, ExcludeOpTypes: opts.ExcludeOpTypes}); reason != "" {
			skip(n, reason)
			continue
		}
		// activation inputs
		var ins []string
		for i, in := range n.Inputs {
			if in == "" || (count >= 0 && i >= count) {
				continue
			}
			ins = append(ins, in)
		}
		bad := ""
		for _, in := range ins {
			if g.Constant(in) != nil {
				bad = fmt.Sprintf("its input %q is a constant, and a constant operand of %s is not quantized here", in, n.OpType)
				break
			}
			if t, ok := g.TypeOf(in); ok && t.ElemType != graph.Float {
				bad = fmt.Sprintf("its input %q is %s, not float32", in, t.ElemType)
				break
			}
			if !known[in] {
				bad = fmt.Sprintf("the calibration gives no range for its input %q", in)
				break
			}
		}
		derived := map[string]bool{}
		if bad == "" {
			for _, out := range n.Outputs {
				if out == "" {
					continue
				}
				if t, ok := g.TypeOf(out); ok && t.ElemType != graph.Float {
					bad = fmt.Sprintf("its output %q is %s, not float32", out, t.ElemType)
					break
				}
				switch {
				case known[out]:
				case n.OpType == "Sigmoid" || n.OpType == "Softmax" || n.OpType == "Tanh":
					derived[out] = true
				case qnnDirectOps[n.OpType] && known[n.Input(0)]:
					derived[out] = true
				default:
					bad = fmt.Sprintf("the calibration gives no range for its output %q", out)
				}
				if bad != "" {
					break
				}
			}
		}
		if bad == "" {
			if _, isConv := qnnConvOps[n.OpType]; isConv {
				w := g.Constant(n.Input(1))
				switch {
				case w == nil:
					bad = fmt.Sprintf("its weight %q is not a constant", n.Input(1))
				case w.External:
					bad = fmt.Sprintf("its weight %q is in external data", w.Name)
				case w.DataType != graph.Float:
					bad = fmt.Sprintf("its weight %q is %s, not float32", w.Name, w.DataType)
				case len(w.Dims) < 3:
					bad = fmt.Sprintf("its weight has shape %v", w.Dims)
				}
				if bad == "" && n.Input(2) != "" {
					b := g.Constant(n.Input(2))
					if b == nil || b.DataType != graph.Float || b.External || len(b.Dims) != 1 {
						bad = fmt.Sprintf("its bias %q is not a constant float32 vector", n.Input(2))
					}
				}
			}
		}
		if bad != "" {
			skip(n, bad)
			continue
		}
		pl.nodes[n] = true
		for out := range derived {
			known[out] = true
		}
		for _, in := range ins {
			wanted[in] = true
		}
		for _, out := range n.Outputs {
			if out != "" {
				wanted[out] = true
			}
		}
	}
	// A node is quantized only when its tensors will all have parameters; the
	// plan keeps graph order for the tensors.
	seen := map[string]bool{}
	add := func(name string) {
		if wanted[name] && !seen[name] {
			seen[name] = true
			pl.order = append(pl.order, name)
		}
	}
	for _, v := range g.Inputs {
		add(v.Name)
	}
	for _, n := range g.Nodes {
		for _, in := range n.Inputs {
			add(in)
		}
		for _, out := range n.Outputs {
			add(out)
		}
	}
	// Parameters are computed in this order; a derived tensor needs its
	// input first, so sort the tensors by the position of their producer.
	pos := map[string]int{}
	for i, v := range g.Inputs {
		pos[v.Name] = -len(g.Inputs) + i
	}
	for i, n := range g.Nodes {
		for _, out := range n.Outputs {
			pos[out] = i
		}
	}
	sort.SliceStable(pl.order, func(i, j int) bool { return pos[pl.order[i]] < pos[pl.order[j]] })
	return pl
}

func (p QParams) typeOr(act QNNType) QNNType {
	if p.Type == "" {
		return act
	}
	return p.Type
}

func validateQParams(name string, p QParams, act QNNType) error {
	t := p.typeOr(act)
	if t != QNNUint8 && t != QNNUint16 {
		return fmt.Errorf("%w: Fixed %q has type %q", ErrRefused, name, p.Type)
	}
	if !(p.Scale > 0) || !finite(p.Scale) {
		return fmt.Errorf("%w: Fixed %q has scale %g, and a scale is positive", ErrRefused, name, p.Scale)
	}
	max := int32(255)
	if t == QNNUint16 {
		max = 65535
	}
	if p.ZeroPoint < 0 || p.ZeroPoint > max {
		return fmt.Errorf("%w: Fixed %q has zero point %d outside 0..%d", ErrRefused, name, p.ZeroPoint, max)
	}
	return nil
}

func finite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

// fixedOutput is the scale and zero point the HTP requires of the output of
// Sigmoid, Softmax and Tanh.
func fixedOutput(op string, act QNNType) QParams {
	switch {
	case op == "Tanh" && act == QNNUint16:
		return QParams{Type: act, Scale: 1.0 / 32768, ZeroPoint: 32768}
	case op == "Tanh":
		return QParams{Type: act, Scale: 1.0 / 128, ZeroPoint: 128}
	case act == QNNUint16:
		return QParams{Type: act, Scale: 1.0 / 65536, ZeroPoint: 0}
	}
	return QParams{Type: act, Scale: 1.0 / 256, ZeroPoint: 0}
}

// rangeParams is the scale and zero point of an asymmetric unsigned
// quantization of a range, in float32 as ONNX Runtime computes them.
func rangeParams(name string, r Range, act QNNType) (QParams, error) {
	qmax := float32(255)
	if act == QNNUint16 {
		qmax = 65535
	}
	lo, hi := r.Min, r.Max
	if lo > 0 {
		lo = 0
	}
	if hi < 0 {
		hi = 0
	}
	if w := lo + MinimumRealRange; hi < w {
		hi = w
	}
	scale := float32(float64(hi-lo) / float64(qmax)) // the width in float32, divided in float64 and rounded once, as ONNX Runtime does
	if !(scale > 0) || !finite(scale) {
		return QParams{}, fmt.Errorf("%w: the range [%g, %g] of %q gives no valid scale", ErrRefused, r.Min, r.Max, name)
	}
	zero := float32(math.RoundToEven(float64(0 - lo/scale)))
	if zero < 0 {
		zero = 0
	}
	if zero > qmax {
		zero = qmax
	}
	return QParams{Type: act, Scale: scale, ZeroPoint: int32(zero)}, nil
}

// zeroTensor is a scalar zero point of the type.
func zeroTensor(name string, t QNNType, v int32) *graph.Tensor {
	if t == QNNUint16 {
		return &graph.Tensor{Name: name, DataType: graph.Uint16, RawData: binary.LittleEndian.AppendUint16(nil, uint16(v))}
	}
	return &graph.Tensor{Name: name, DataType: graph.Uint8, RawData: []byte{byte(v)}}
}

// quantizeWeightUint8 quantizes a weight per channel along an axis into uint8
// with zero point 128. The scale is 2*max|w|/255 per channel, or fixed (every
// channel) when fixed is not zero. A channel of zeros gets scale 1.
func quantizeWeightUint8(values []float32, dims []int64, axis int, fixed float32) ([]uint8, []float32, float32) {
	channels := int(dims[axis])
	inner := 1
	for _, d := range dims[axis+1:] {
		inner *= int(d)
	}
	channelOf := func(i int) int { return (i / inner) % channels }
	peak := make([]float32, channels)
	for i, v := range values {
		c := channelOf(i)
		a := float32(math.Abs(float64(v)))
		if a > peak[c] {
			peak[c] = a
		}
	}
	scales := make([]float32, channels)
	for c, p := range peak {
		switch {
		case fixed > 0:
			scales[c] = fixed
		case p == 0:
			scales[c] = 1
		default:
			scales[c] = 2 * p / 255
		}
	}
	q := make([]uint8, len(values))
	var maxErr float64
	for i, v := range values {
		s := scales[channelOf(i)]
		r := math.RoundToEven(float64(v/s)) + 128 // float32 division, as ONNX Runtime computes it
		r = math.Max(0, math.Min(255, r))
		q[i] = uint8(r)
		maxErr = math.Max(maxErr, math.Abs(float64(v)-(r-128)*float64(s)))
	}
	return q, scales, float32(maxErr)
}
