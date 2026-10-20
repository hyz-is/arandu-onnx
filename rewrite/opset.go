package rewrite

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/hyz-is/arandu-onnx/graph"
)

// The default-domain opsets UpgradeOpset reads and writes.
const (
	// OpsetUpgradeTarget is the opset UpgradeOpset writes: per-channel
	// DequantizeLinear, which quantize.QDQ needs, is opset 13.
	OpsetUpgradeTarget = 13
	// OpsetUpgradeMinimum is the oldest opset UpgradeOpset reads. Below 7,
	// the arithmetic operators broadcast by attribute and BatchNormalization
	// and Dropout carried is_test, which are not rewritten here.
	OpsetUpgradeMinimum = 7
)

// OpsetOptions says what UpgradeOpset writes.
type OpsetOptions struct {
	// Target is the default-domain opset to write. Zero means
	// OpsetUpgradeTarget, which is also the only target this version
	// writes: the operator changes are enumerated up to it.
	Target int64
}

// OpsetConversion is one node whose operator changed signature or meaning
// between the model's opset and the target, and what replaced it.
type OpsetConversion struct {
	// Node and OpType are the node's name and operator.
	Node   string `json:"node"`
	OpType string `json:"op_type"`
	// Change says what was rewritten, such as "axes attribute to input".
	Change string `json:"change"`
	// Nodes is the names of the nodes that replaced it, the node itself
	// included when it was kept.
	Nodes []string `json:"nodes"`
}

// OpsetReport is what UpgradeOpset did.
type OpsetReport struct {
	// OpsetBefore and OpsetAfter are the default-domain opset.
	OpsetBefore int64 `json:"opset_before"`
	OpsetAfter  int64 `json:"opset_after"`
	// IRBefore and IRAfter are the IR version; opset 13 needs IR 7.
	IRBefore int64 `json:"ir_before"`
	IRAfter  int64 `json:"ir_after"`
	// NodesBefore and NodesAfter are the graph's node counts.
	NodesBefore int `json:"nodes_before"`
	NodesAfter  int `json:"nodes_after"`
	// Converted is every node rewritten, in graph order.
	Converted []OpsetConversion `json:"converted"`
	// Unchanged is "Op:count" for the operators whose definition at the
	// target computes the same as at the model's opset, kept as they were.
	Unchanged []string `json:"unchanged"`
}

// opsetUnchanged is every default-domain operator whose definition does not
// change between opset 7 and 13 in a way that changes what a valid node of
// opset 7 or later computes. From the ONNX operator changelog: the versions
// between are new element types (bfloat16 at 13, integers and strings at 8,
// 9 and 12), negative axes and indices accepted (11), multidirectional
// broadcasting for Max, Min, Sum and Mean (8, a superset), optional inputs
// (Gemm's C at 11), new attributes whose default is the old behaviour
// (ceil_mode and dilations of the pools at 10, DepthToSpace's mode at 11,
// TopK's largest and sorted at 11, ArgMax's select_last_index at 12), and
// wording. Conv and LpPool are here with a guard on auto_pad, whose SAME
// padding was redefined at 11 for strides above 1, and ConvTranspose with a
// guard on any SAME padding, whose odd remainder changed sides at 11.
var opsetUnchanged = map[string]bool{
	"Abs": true, "Acos": true, "Acosh": true, "Add": true, "And": true, "ArgMax": true, "ArgMin": true,
	"Asin": true, "Asinh": true, "Atan": true, "Atanh": true, "AveragePool": true, "Cast": true,
	"Ceil": true, "Concat": true, "Constant": true, "ConstantOfShape": true, "Conv": true, "ConvTranspose": true, "Cos": true,
	"Cosh": true, "DepthToSpace": true, "Div": true, "Elu": true, "Equal": true, "Erf": true, "Exp": true,
	"Expand": true, "Flatten": true, "Floor": true, "Gather": true, "Gemm": true,
	"GlobalAveragePool": true, "GlobalLpPool": true, "GlobalMaxPool": true, "Greater": true,
	"HardSigmoid": true, "Identity": true, "InstanceNormalization": true, "LRN": true,
	"LeakyRelu": true, "Less": true, "Log": true, "LpPool": true, "MatMul": true, "Max": true,
	"MaxPool": true, "Mean": true, "Min": true, "Mul": true, "Neg": true, "Not": true, "Or": true,
	"PRelu": true, "Pow": true, "Reciprocal": true, "ReduceL1": true, "ReduceL2": true,
	"ReduceLogSum": true, "ReduceLogSumExp": true, "ReduceMax": true, "ReduceMean": true,
	"ReduceMin": true, "ReduceProd": true, "ReduceSumSquare": true, "Relu": true, "Reshape": true,
	"Selu": true, "Shape": true, "Sigmoid": true, "Sign": true, "Sin": true, "Sinh": true, "Size": true,
	"Softplus": true, "Softsign": true, "SpaceToDepth": true, "Sqrt": true, "Sub": true, "Sum": true,
	"Tan": true, "Tanh": true, "Tile": true, "Transpose": true, "Where": true, "Xor": true,
}

// opsetConverted is, for each operator UpgradeOpset rewrites, the opset at
// which its signature or meaning changed: a model below it gets the node
// rewritten. Upsample is deprecated (10) and always becomes Resize.
var opsetConverted = map[string]int64{
	"BatchNormalization": 9,
	"Slice":              10,
	"TopK":               10,
	"Clip":               11,
	"Pad":                11,
	"Resize":             11,
	"Dropout":            12,
	"Squeeze":            13,
	"Unsqueeze":          13,
	"Split":              13,
	"ReduceSum":          13,
	"Softmax":            13,
	"LogSoftmax":         13,
	"Hardmax":            13,
	"Upsample":           math.MaxInt64,
}

// UpgradeOpset raises the model's default-domain opset to 13, rewriting every
// node whose operator changed signature or meaning in between so that the
// model computes what it did.
//
// The rewrites, from the ONNX operator changelog:
//
//   - BatchNormalization (9): the spatial attribute is removed; spatial = 0
//     is refused.
//   - Slice (10): starts, ends and axes become int64 inputs.
//   - TopK (10): k becomes the int64 input K.
//   - Clip (11): min and max become inputs of the input's element type,
//     which has to be known to be float32 when they are set.
//   - Pad (11): pads becomes an int64 input and value the input
//     constant_value, in float32 as Clip's.
//   - Resize-10 (11): an empty roi input is added, and the behaviour of
//     opset 10 is written out: coordinate_transformation_mode "asymmetric"
//     and, for nearest, nearest_mode "floor" when every scale is at least 1
//     and "ceil" when every scale is at most 1, which needs constant scales.
//   - Upsample (deprecated at 10): becomes Resize the same way, its scales
//     attribute (opset 7 and 8) an input.
//   - Dropout (12): ratio becomes an input; a mask output that something
//     reads is refused below 10, where it was float rather than bool.
//   - Squeeze, Unsqueeze, ReduceSum (13): axes becomes an int64 input. Split
//     (13): split becomes an int64 input.
//   - Softmax, LogSoftmax, Hardmax (13): below 13 they coerce the input to 2-D
//     at axis; at 13 they work along one axis. A node whose axis is the last
//     dimension, which is what infer can tell from the graph, keeps the node
//     with axis written explicitly; any other becomes Flatten, the operator
//     along axis 1, and Reshape back to the input's shape, which is the old
//     definition exactly.
//
// Every other standard operator has to be one that computes the same at 13
// (opsetUnchanged); Conv and LpPool with auto_pad SAME and a stride above 1,
// and ConvTranspose with auto_pad SAME, are refused below 11, where SAME was
// defined differently. A standard operator in neither list, a model below
// opset 7, a model with local functions or subgraphs, and a target other
// than 13 are refused with ErrUnsupported, and the model is left unchanged.
// Operators of other domains keep their own opsets and are not touched. A
// model already at 13 or later is left unchanged and reported so. The IR
// version is raised to 7, which opset 13 requires; the constants the
// rewrites add are initializers.
func UpgradeOpset(m *graph.Model, opts OpsetOptions) (OpsetReport, error) {
	if m == nil || m.Graph == nil {
		return OpsetReport{}, fmt.Errorf("%w: the model has no graph", ErrUnsupported)
	}
	target := opts.Target
	if target == 0 {
		target = OpsetUpgradeTarget
	}
	if target != OpsetUpgradeTarget {
		return OpsetReport{}, fmt.Errorf("%w: opset %d is not a target: this version enumerates the operator changes up to opset %d and writes that one", ErrUnsupported, target, OpsetUpgradeTarget)
	}
	from, ok := m.Opset(graph.DefaultDomain)
	if !ok {
		return OpsetReport{}, fmt.Errorf("%w: the model imports no default-domain opset", ErrUnsupported)
	}
	g := m.Graph
	report := OpsetReport{OpsetBefore: from, OpsetAfter: from, IRBefore: m.IRVersion, IRAfter: m.IRVersion, NodesBefore: len(g.Nodes), NodesAfter: len(g.Nodes)}
	if from >= target {
		report.Unchanged = g.SortedOpCounts()
		return report, nil
	}
	if from < OpsetUpgradeMinimum {
		return OpsetReport{}, fmt.Errorf("%w: the model is at opset %d, and the upgrade reads opset %d or later", ErrUnsupported, from, OpsetUpgradeMinimum)
	}
	if m.HasFunctions() || g.HasSubgraphs() {
		return OpsetReport{}, fmt.Errorf("%w: the model has local functions or subgraphs, whose bodies are not read, so their operators cannot be upgraded", ErrUnsupported)
	}

	u := &upgrader{g: g, from: from, namer: graph.NewNamer(g), info: infer(g)}
	var unsupported []string
	seen := map[string]bool{}
	replaced := map[*graph.Node][]*graph.Node{}
	unchanged := map[string]int{}
	for _, n := range g.Nodes {
		if !n.IsStandard() {
			continue
		}
		at, converted := opsetConverted[n.OpType]
		switch {
		case converted && from < at:
			nodes, change, err := u.convert(n)
			if err != nil {
				return OpsetReport{}, err
			}
			replaced[n] = nodes
			names := make([]string, len(nodes))
			for i, r := range nodes {
				names[i] = r.Name
			}
			report.Converted = append(report.Converted, OpsetConversion{Node: n.Name, OpType: n.OpType, Change: change, Nodes: names})
		case converted || opsetUnchanged[n.OpType]:
			if err := u.guard(n); err != nil {
				return OpsetReport{}, err
			}
			unchanged[n.OpType]++
		default:
			if !seen[n.OpType] {
				seen[n.OpType] = true
				unsupported = append(unsupported, n.OpType)
			}
		}
	}
	if len(unsupported) > 0 {
		sort.Strings(unsupported)
		return OpsetReport{}, fmt.Errorf("%w: the upgrade from opset %d to %d does not handle %s: their changes between those opsets are not enumerated here, so the model is left as it is", ErrUnsupported, from, target, strings.Join(unsupported, ", "))
	}

	// Every node is planned: from here on the model is changed.
	nodes := make([]*graph.Node, 0, len(g.Nodes)+len(u.initializers))
	for _, n := range g.Nodes {
		if with, ok := replaced[n]; ok {
			nodes = append(nodes, with...)
			continue
		}
		nodes = append(nodes, n)
	}
	g.Nodes = nodes
	g.Initializers = append(g.Initializers, u.initializers...)
	m.SetOpset(graph.DefaultDomain, target)
	if m.IRVersion < 7 {
		m.IRVersion = 7
	}
	if err := g.CheckTopological(); err != nil {
		return OpsetReport{}, fmt.Errorf("rewrite: the upgraded graph is out of order, which is a defect of the upgrade: %w", err)
	}
	report.OpsetAfter = target
	report.IRAfter = m.IRVersion
	report.NodesAfter = len(g.Nodes)
	report.Unchanged = countStrings(unchanged)
	return report, nil
}

// upgrader plans the rewrites of one UpgradeOpset run without changing the
// graph: the nodes it returns are new values, and the initializers it adds
// are kept aside until every node is planned.
type upgrader struct {
	g            *graph.Graph
	from         int64
	namer        *graph.Namer
	info         map[string]inferred
	initializers []*graph.Tensor
}

func (u *upgrader) fail(n *graph.Node, format string, args ...any) error {
	return fmt.Errorf("%w: %s %q at opset %d: %s", ErrUnsupported, n.OpType, n.Name, u.from, fmt.Sprintf(format, args...))
}

// ints adds an int64 vector initializer and returns its name.
func (u *upgrader) ints(n *graph.Node, what string, values []int64) string {
	name := u.namer.Fresh(nodeBase(n) + "/" + what)
	u.initializers = append(u.initializers, graph.Int64Tensor(name, []int64{int64(len(values))}, values))
	return name
}

// floats adds a float32 initializer, a scalar when scalar is set, and
// returns its name.
func (u *upgrader) floats(n *graph.Node, what string, values []float32, scalar bool) string {
	name := u.namer.Fresh(nodeBase(n) + "/" + what)
	var dims []int64
	if !scalar {
		dims = []int64{int64(len(values))}
	}
	u.initializers = append(u.initializers, graph.Float32Tensor(name, dims, values))
	return name
}

// nodeBase is a stem for the names of what a rewrite adds for a node.
func nodeBase(n *graph.Node) string {
	if n.Name != "" {
		return n.Name
	}
	return n.OpType + "_" + n.Outputs[0]
}

// guard refuses a node kept as it is whose attributes changed meaning.
func (u *upgrader) guard(n *graph.Node) error {
	if u.from >= 11 {
		return nil
	}
	pad := string(attrString(n, "auto_pad", "NOTSET"))
	if pad != "SAME_UPPER" && pad != "SAME_LOWER" {
		return nil
	}
	switch n.OpType {
	case "ConvTranspose":
		return u.fail(n, "auto_pad %s was redefined at opset 11 (the side the odd padding goes to was swapped); give it explicit pads", pad)
	case "Conv", "LpPool":
		for _, s := range n.IntsOr("strides", nil) {
			if s != 1 {
				return u.fail(n, "auto_pad %s with strides %v was redefined at opset 11; give it explicit pads", pad, n.IntsOr("strides", nil))
			}
		}
	}
	return nil
}

// convert plans the rewrite of one node whose operator changed.
func (u *upgrader) convert(n *graph.Node) ([]*graph.Node, string, error) {
	c := copyNode(n)
	switch n.OpType {
	case "BatchNormalization":
		if a := n.Attribute("spatial"); a != nil && a.I != 1 {
			return nil, "", u.fail(n, "spatial = %d (per-activation statistics) was removed at opset 9", a.I)
		}
		c.Attributes = withoutAttributes(c.Attributes, "spatial")
		return []*graph.Node{c}, "spatial attribute removed", nil

	case "Slice":
		starts, ends := n.Attribute("starts"), n.Attribute("ends")
		if starts == nil || ends == nil || len(starts.Ints) != len(ends.Ints) {
			return nil, "", u.fail(n, "starts and ends are required and of the same length")
		}
		c.Inputs = []string{n.Input(0), u.ints(n, "starts", starts.Ints), u.ints(n, "ends", ends.Ints)}
		if axes := n.Attribute("axes"); axes != nil {
			c.Inputs = append(c.Inputs, u.ints(n, "axes", axes.Ints))
		}
		c.Attributes = withoutAttributes(c.Attributes, "starts", "ends", "axes")
		return []*graph.Node{c}, "starts, ends and axes attributes to inputs", nil

	case "TopK":
		k := n.Attribute("k")
		if k == nil {
			return nil, "", u.fail(n, "the k attribute is required below opset 10")
		}
		c.Inputs = []string{n.Input(0), u.ints(n, "k", []int64{k.I})}
		c.Attributes = withoutAttributes(c.Attributes, "k")
		return []*graph.Node{c}, "k attribute to input K", nil

	case "Clip":
		lo, hi := n.Attribute("min"), n.Attribute("max")
		if lo == nil && hi == nil {
			return []*graph.Node{c}, "no min or max: inputs left out", nil
		}
		if err := u.needFloat32(n, "min and max"); err != nil {
			return nil, "", err
		}
		c.Inputs = []string{n.Input(0), "", ""}
		if lo != nil {
			c.Inputs[1] = u.floats(n, "min", []float32{lo.F}, true)
		}
		if hi != nil {
			c.Inputs[2] = u.floats(n, "max", []float32{hi.F}, true)
		}
		c.Inputs = trimOptional(c.Inputs)
		c.Attributes = withoutAttributes(c.Attributes, "min", "max")
		return []*graph.Node{c}, "min and max attributes to inputs", nil

	case "Pad":
		pads := n.Attribute("pads")
		if pads == nil {
			return nil, "", u.fail(n, "the pads attribute is required below opset 11")
		}
		c.Inputs = []string{n.Input(0), u.ints(n, "pads", pads.Ints)}
		mode := string(attrString(n, "mode", "constant"))
		if v := n.Attribute("value"); v != nil && v.F != 0 && mode == "constant" {
			if err := u.needFloat32(n, "constant_value"); err != nil {
				return nil, "", err
			}
			c.Inputs = append(c.Inputs, u.floats(n, "constant_value", []float32{v.F}, true))
		}
		c.Attributes = withoutAttributes(c.Attributes, "pads", "value")
		return []*graph.Node{c}, "pads and value attributes to inputs", nil

	case "Resize", "Upsample":
		return u.resize(n, c)

	case "Dropout":
		if u.from < 10 && len(n.Outputs) > 1 && n.Outputs[1] != "" && (u.g.Used(n.Outputs[1])) {
			return nil, "", u.fail(n, "its mask output %q is read, and the mask was float below opset 10 and bool from 10 on", n.Outputs[1])
		}
		c.Inputs = []string{n.Input(0)}
		if r := n.Attribute("ratio"); r != nil {
			c.Inputs = append(c.Inputs, u.floats(n, "ratio", []float32{r.F}, true))
		}
		c.Attributes = withoutAttributes(c.Attributes, "ratio")
		return []*graph.Node{c}, "ratio attribute to input (inference mode, as before)", nil

	case "Squeeze", "Unsqueeze", "ReduceSum":
		axes := n.Attribute("axes")
		if axes == nil {
			if n.OpType == "Unsqueeze" {
				return nil, "", u.fail(n, "the axes attribute is required below opset 13")
			}
			return []*graph.Node{c}, "no axes: input left out", nil
		}
		c.Inputs = []string{n.Input(0), u.ints(n, "axes", axes.Ints)}
		c.Attributes = withoutAttributes(c.Attributes, "axes")
		return []*graph.Node{c}, "axes attribute to input", nil

	case "Split":
		split := n.Attribute("split")
		if split == nil {
			return []*graph.Node{c}, "no split: equal parts, input left out", nil
		}
		c.Inputs = []string{n.Input(0), u.ints(n, "split", split.Ints)}
		c.Attributes = withoutAttributes(c.Attributes, "split")
		return []*graph.Node{c}, "split attribute to input", nil

	case "Softmax", "LogSoftmax", "Hardmax":
		return u.softmax(n, c)
	}
	return nil, "", u.fail(n, "no rewrite is written for it, which is a defect of the upgrade")
}

// needFloat32 refuses a rewrite that has to write a value in the element
// type of the node's first input when that type is not known to be float32.
func (u *upgrader) needFloat32(n *graph.Node, what string) error {
	if elem := u.info[n.Input(0)].elem; elem != graph.Float {
		return u.fail(n, "%s become inputs of the type of %q, which the graph records as %s; only float32 is written here", what, n.Input(0), elem)
	}
	return nil
}

// resize plans Resize-10 and Upsample as Resize-13.
func (u *upgrader) resize(n, c *graph.Node) ([]*graph.Node, string, error) {
	mode := string(attrString(n, "mode", "nearest"))
	if mode != "nearest" && mode != "linear" {
		return nil, "", u.fail(n, "mode %q is not nearest or linear", mode)
	}
	for _, a := range n.Attributes {
		if a.Name != "mode" && a.Name != "scales" {
			return nil, "", u.fail(n, "attribute %q is not one it defines", a.Name)
		}
	}
	var scales []float32
	scalesName := n.Input(1)
	if a := n.Attribute("scales"); a != nil && n.OpType == "Upsample" && u.from < 9 {
		scales = a.Floats
		scalesName = u.floats(n, "scales", scales, false)
	} else if scalesName == "" {
		return nil, "", u.fail(n, "it has no scales")
	} else if t := u.g.Constant(scalesName); t != nil {
		vs, err := t.Float32s()
		if err != nil {
			return nil, "", u.fail(n, "scales: %v", err)
		}
		scales = vs
	}

	c.OpType = "Resize"
	c.Inputs = []string{n.Input(0), "", scalesName}
	c.Attributes = []*graph.Attribute{
		graph.StringAttribute("coordinate_transformation_mode", "asymmetric"),
		graph.StringAttribute("mode", mode),
	}
	change := "Resize-13 with an empty roi, coordinate_transformation_mode asymmetric"
	if n.OpType == "Upsample" {
		change = "Upsample to " + change
	}
	if mode == "nearest" {
		up, down := false, false
		for _, s := range scales {
			up = up || s > 1
			down = down || s < 1
		}
		switch {
		case scales == nil && n.OpType == "Upsample":
			// Upsample's scales are at least 1 by definition.
			up = true
		case scales == nil:
			return nil, "", u.fail(n, "nearest rounding at opset 10 depends on whether each scale is above or below 1, and its scales %q are computed at run time", scalesName)
		case up && down:
			return nil, "", u.fail(n, "scales %v enlarge one axis and shrink another, and opset 10 rounded each axis its own way, which one nearest_mode cannot write", scales)
		}
		nearest := "floor"
		if down {
			nearest = "ceil"
		}
		c.Attributes = append(c.Attributes, graph.StringAttribute("nearest_mode", nearest))
		change += ", nearest_mode " + nearest
	}
	return []*graph.Node{c}, change, nil
}

// softmax plans Softmax, LogSoftmax and Hardmax of opset 1 or 11 as opset 13.
func (u *upgrader) softmax(n, c *graph.Node) ([]*graph.Node, string, error) {
	axis := n.Int("axis", 1)
	rank := u.info[n.Input(0)].rank
	last := axis == -1
	if rank >= 0 {
		if axis < -int64(rank) || axis >= int64(rank) {
			return nil, "", u.fail(n, "axis %d is outside a tensor of rank %d", axis, rank)
		}
		norm := axis
		if norm < 0 {
			norm += int64(rank)
		}
		last = norm == int64(rank-1)
		if last {
			axis = norm
		}
	}
	if last {
		c.SetAttribute(graph.IntAttribute("axis", axis))
		return []*graph.Node{c}, fmt.Sprintf("axis %d is the last dimension: the same along one axis, axis written explicitly", axis), nil
	}

	base := nodeBase(n)
	shape := u.namer.Fresh(n.Input(0) + "/shape")
	flat := u.namer.Fresh(n.Input(0) + "/flattened")
	result := u.namer.Fresh(n.Outputs[0] + "/flattened")
	shapeNode := &graph.Node{Name: u.namer.Fresh(base + "/Shape"), OpType: "Shape", Inputs: []string{n.Input(0)}, Outputs: []string{shape}}
	flatten := &graph.Node{
		Name: u.namer.Fresh(base + "/Flatten"), OpType: "Flatten", Inputs: []string{n.Input(0)}, Outputs: []string{flat},
		Attributes: []*graph.Attribute{graph.IntAttribute("axis", axis)},
	}
	c.Inputs = []string{flat}
	c.Outputs = []string{result}
	c.SetAttribute(graph.IntAttribute("axis", 1))
	reshape := &graph.Node{Name: u.namer.Fresh(base + "/Reshape"), OpType: "Reshape", Inputs: []string{result, shape}, Outputs: []string{n.Outputs[0]}}
	what := "rank unknown"
	if rank >= 0 {
		what = fmt.Sprintf("rank %d", rank)
	}
	return []*graph.Node{shapeNode, flatten, c, reshape}, fmt.Sprintf("axis %d (%s) coerced to 2-D as before: Flatten, %s along axis 1, Reshape", n.Int("axis", 1), what, n.OpType), nil
}

// copyNode returns a node that can be changed without changing n.
func copyNode(n *graph.Node) *graph.Node {
	c := *n
	c.Inputs = append([]string(nil), n.Inputs...)
	c.Outputs = append([]string(nil), n.Outputs...)
	c.Attributes = append([]*graph.Attribute(nil), n.Attributes...)
	return &c
}

// withoutAttributes returns the attributes not named.
func withoutAttributes(attrs []*graph.Attribute, names ...string) []*graph.Attribute {
	out := attrs[:0:0]
	for _, a := range attrs {
		drop := false
		for _, name := range names {
			if a.Name == name {
				drop = true
			}
		}
		if !drop {
			out = append(out, a)
		}
	}
	return out
}

// attrString is a string attribute, or def.
func attrString(n *graph.Node, name, def string) []byte {
	if a := n.Attribute(name); a != nil {
		return a.S
	}
	return []byte(def)
}

// trimOptional drops the trailing empty names of optional inputs.
func trimOptional(inputs []string) []string {
	for len(inputs) > 0 && inputs[len(inputs)-1] == "" {
		inputs = inputs[:len(inputs)-1]
	}
	return inputs
}

// countStrings is a count map as sorted "key:count" strings.
func countStrings(counts map[string]int) []string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, len(keys))
	for i, k := range keys {
		out[i] = fmt.Sprintf("%s:%d", k, counts[k])
	}
	return out
}
