package graph

import (
	"fmt"
	"sort"
	"strconv"
)

// Attribute returns the node's attribute of that name, or nil.
func (n *Node) Attribute(name string) *Attribute {
	for _, a := range n.Attributes {
		if a.Name == name {
			return a
		}
	}
	return nil
}

// Int returns an int attribute, or def when the node has none of that name.
func (n *Node) Int(name string, def int64) int64 {
	if a := n.Attribute(name); a != nil {
		return a.I
	}
	return def
}

// IntsOr returns an ints attribute, or def when the node has none of that
// name.
func (n *Node) IntsOr(name string, def []int64) []int64 {
	if a := n.Attribute(name); a != nil {
		return append([]int64(nil), a.Ints...)
	}
	return def
}

// SetAttribute replaces the node's attribute of the same name, or adds it.
func (n *Node) SetAttribute(a *Attribute) {
	for i, existing := range n.Attributes {
		if existing.Name == a.Name {
			n.Attributes[i] = a
			return
		}
	}
	n.Attributes = append(n.Attributes, a)
}

// IsStandard reports whether the node is a standard ONNX operator, in the
// domain "" or "ai.onnx".
func (n *Node) IsStandard() bool { return isDefaultDomain(n.Domain) }

// HasSubgraph reports whether one of the node's attributes is a graph, as the
// bodies of If, Loop and Scan are. Subgraphs are kept as they were read and
// never rewritten.
func (n *Node) HasSubgraph() bool {
	for _, a := range n.Attributes {
		if a.subgraph {
			return true
		}
	}
	return false
}

// Input returns the node's i-th input, or "" when it has fewer.
func (n *Node) Input(i int) string {
	if i < len(n.Inputs) {
		return n.Inputs[i]
	}
	return ""
}

// IntAttribute returns an int attribute.
func IntAttribute(name string, v int64) *Attribute {
	return &Attribute{Name: name, Type: AttributeInt, I: v}
}

// IntsAttribute returns an ints attribute.
func IntsAttribute(name string, vs []int64) *Attribute {
	return &Attribute{Name: name, Type: AttributeInts, Ints: append([]int64{}, vs...)}
}

// FloatAttribute returns a float attribute.
func FloatAttribute(name string, v float32) *Attribute {
	return &Attribute{Name: name, Type: AttributeFloat, F: v}
}

// StringAttribute returns a string attribute.
func StringAttribute(name, v string) *Attribute {
	return &Attribute{Name: name, Type: AttributeString, S: []byte(v)}
}

// TensorAttribute returns a tensor attribute.
func TensorAttribute(name string, t *Tensor) *Attribute {
	return &Attribute{Name: name, Type: AttributeTensor, T: t}
}

// Initializer returns the initializer of that name, or nil.
func (g *Graph) Initializer(name string) *Tensor {
	for _, t := range g.Initializers {
		if t.Name == name {
			return t
		}
	}
	return nil
}

// Constant returns the constant value of a tensor: an initializer that is
// not also a graph input (which a caller could override), or the value of
// a Constant node. It returns nil for a tensor computed at run time.
func (g *Graph) Constant(name string) *Tensor {
	if name == "" {
		return nil
	}
	if t := g.Initializer(name); t != nil {
		for _, in := range g.Inputs {
			if in.Name == name {
				return nil
			}
		}
		return t
	}
	n := g.Producer(name)
	if n == nil || n.OpType != "Constant" || !n.IsStandard() || len(n.Attributes) != 1 {
		return nil
	}
	a := n.Attributes[0]
	switch a.Name {
	case "value":
		if a.T == nil {
			return nil
		}
		t := *a.T
		t.Name = name
		return &t
	case "value_int":
		return Int64Tensor(name, nil, []int64{a.I})
	case "value_ints":
		return Int64Tensor(name, []int64{int64(len(a.Ints))}, a.Ints)
	case "value_float":
		return Float32Tensor(name, nil, []float32{a.F})
	case "value_floats":
		return Float32Tensor(name, []int64{int64(len(a.Floats))}, a.Floats)
	}
	return nil
}

// Producer returns the node that writes a tensor, or nil for a graph input,
// an initializer or a name nothing writes.
func (g *Graph) Producer(name string) *Node {
	if name == "" {
		return nil
	}
	for _, n := range g.Nodes {
		for _, out := range n.Outputs {
			if out == name {
				return n
			}
		}
	}
	return nil
}

// Consumers returns the nodes that read a tensor, in graph order, each once.
func (g *Graph) Consumers(name string) []*Node {
	if name == "" {
		return nil
	}
	var out []*Node
	for _, n := range g.Nodes {
		for _, in := range n.Inputs {
			if in == name {
				out = append(out, n)
				break
			}
		}
	}
	return out
}

// IsOutput reports whether a tensor is a graph output.
func (g *Graph) IsOutput(name string) bool {
	for _, v := range g.Outputs {
		if v.Name == name {
			return true
		}
	}
	return false
}

// IsInput reports whether a tensor is a graph input.
func (g *Graph) IsInput(name string) bool {
	for _, v := range g.Inputs {
		if v.Name == name {
			return true
		}
	}
	return false
}

// TypeOf returns the tensor type the graph records for a tensor, from its
// inputs, outputs, value info or initializers, and false when it records
// none.
func (g *Graph) TypeOf(name string) (TensorType, bool) {
	for _, list := range [][]*ValueInfo{g.Inputs, g.Outputs, g.ValueInfo} {
		for _, v := range list {
			if v.Name == name {
				if t, ok := v.Tensor(); ok {
					return t, true
				}
			}
		}
	}
	if t := g.Initializer(name); t != nil {
		shape := make([]Dim, len(t.Dims))
		for i, d := range t.Dims {
			shape[i] = Dim{Value: d}
		}
		return TensorType{ElemType: t.DataType, Shape: shape, Ranked: true}, true
	}
	return TensorType{}, false
}

// RemoveInitializer removes an initializer, and the graph input of the same
// name that models before IR version 4 list beside it.
func (g *Graph) RemoveInitializer(name string) {
	kept := g.Initializers[:0]
	for _, t := range g.Initializers {
		if t.Name != name {
			kept = append(kept, t)
		}
	}
	g.Initializers = kept
	inputs := g.Inputs[:0]
	for _, v := range g.Inputs {
		if v.Name != name {
			inputs = append(inputs, v)
		}
	}
	g.Inputs = inputs
}

// Used reports whether any node of the top-level graph reads a tensor, or
// the graph outputs it. Subgraphs are not read: a tensor a subgraph captures
// from the outer scope is reported unused, so callers that remove what is
// unused check HasSubgraphs first.
func (g *Graph) Used(name string) bool {
	if g.IsOutput(name) {
		return true
	}
	for _, n := range g.Nodes {
		for _, in := range n.Inputs {
			if in == name {
				return true
			}
		}
	}
	return false
}

// HasSubgraphs reports whether a node of the graph carries a subgraph.
func (g *Graph) HasSubgraphs() bool {
	for _, n := range g.Nodes {
		if n.HasSubgraph() {
			return true
		}
	}
	return false
}

// OpCounts returns the number of nodes of each operator.
func (g *Graph) OpCounts() map[string]int {
	out := map[string]int{}
	for _, n := range g.Nodes {
		out[n.OpType]++
	}
	return out
}

// Namer hands out tensor and node names no part of a graph already uses.
type Namer struct {
	used map[string]bool
}

// NewNamer returns a Namer over every name the graph uses: node names,
// tensor names and initializer names.
func NewNamer(g *Graph) *Namer {
	used := map[string]bool{"": true}
	for _, n := range g.Nodes {
		used[n.Name] = true
		for _, s := range n.Inputs {
			used[s] = true
		}
		for _, s := range n.Outputs {
			used[s] = true
		}
	}
	for _, t := range g.Initializers {
		used[t.Name] = true
	}
	for _, list := range [][]*ValueInfo{g.Inputs, g.Outputs, g.ValueInfo} {
		for _, v := range list {
			used[v.Name] = true
		}
	}
	return &Namer{used: used}
}

// Fresh returns base, or base with a numeric suffix, whichever no name in the
// graph or handed out before is, and reserves it.
func (n *Namer) Fresh(base string) string {
	name := base
	for i := 1; n.used[name]; i++ {
		name = base + "_" + strconv.Itoa(i)
	}
	n.used[name] = true
	return name
}

// CheckTopological reports the first node that reads a tensor no earlier node
// writes and that is neither a graph input nor an initializer. It is the
// order ONNX requires, and the check every rewrite here runs on its result.
func (g *Graph) CheckTopological() error {
	known := map[string]bool{"": true}
	for _, v := range g.Inputs {
		known[v.Name] = true
	}
	for _, t := range g.Initializers {
		known[t.Name] = true
	}
	for i, n := range g.Nodes {
		for _, in := range n.Inputs {
			if !known[in] {
				return fmt.Errorf("%w: node %d (%s %q) reads %q before anything writes it", ErrMalformed, i, n.OpType, n.Name, in)
			}
		}
		for _, out := range n.Outputs {
			known[out] = true
		}
	}
	return nil
}

// SortedOpCounts returns OpCounts as "Op:count" strings in operator order,
// for reports.
func (g *Graph) SortedOpCounts() []string {
	counts := g.OpCounts()
	ops := make([]string, 0, len(counts))
	for op := range counts {
		ops = append(ops, op)
	}
	sort.Strings(ops)
	out := make([]string, len(ops))
	for i, op := range ops {
		out[i] = op + ":" + strconv.Itoa(counts[op])
	}
	return out
}
