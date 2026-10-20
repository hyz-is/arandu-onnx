package rewrite

import (
	"fmt"

	"github.com/hyz-is/arandu-onnx/graph"
)

// InitializersReport is what InitializersOutOfInputs did.
type InitializersReport struct {
	// Removed is the graph inputs dropped because an initializer of the same
	// name gives their value, in graph order.
	Removed []string `json:"removed"`
	// InputsBefore and InputsAfter are the graph's input counts.
	InputsBefore int `json:"inputs_before"`
	InputsAfter  int `json:"inputs_after"`
	// IRBefore and IRAfter are the IR version; an initializer outside the
	// inputs needs IR 4.
	IRBefore int64 `json:"ir_before"`
	IRAfter  int64 `json:"ir_after"`
}

// InitializersOutOfInputs drops every graph input that has an initializer of
// the same name, so that the initializer is a constant of the graph.
//
// Before IR version 4 every initializer had to be listed as a graph input,
// and an exporter that keeps doing so writes a model whose weights a caller
// may override at run time: graph.Graph.Constant, quantize.QDQ and the
// constant folding of ONNX Runtime and TensorRT then treat every weight as
// an activation. From IR 4 on, an initializer that is not an input is a
// constant, and that is what this rewrite declares: the model computes the
// same for every input it is given without overriding a weight, and a caller
// can no longer override one. The IR version is raised to 4 when it is below.
// A graph input without an initializer is kept, and so is the order of the
// inputs left.
func InitializersOutOfInputs(m *graph.Model) (InitializersReport, error) {
	if m == nil || m.Graph == nil {
		return InitializersReport{}, fmt.Errorf("%w: the model has no graph", ErrUnsupported)
	}
	g := m.Graph
	report := InitializersReport{InputsBefore: len(g.Inputs), IRBefore: m.IRVersion, IRAfter: m.IRVersion}
	initializers := map[string]bool{}
	for _, t := range g.Initializers {
		initializers[t.Name] = true
	}
	kept := make([]*graph.ValueInfo, 0, len(g.Inputs))
	for _, in := range g.Inputs {
		if initializers[in.Name] {
			report.Removed = append(report.Removed, in.Name)
			continue
		}
		kept = append(kept, in)
	}
	if len(report.Removed) > 0 {
		if len(kept) == 0 {
			return InitializersReport{}, fmt.Errorf("%w: every graph input has an initializer, and a graph with no input left takes nothing from its caller", ErrUnsupported)
		}
		g.Inputs = kept
		if m.IRVersion < 4 {
			m.IRVersion = 4
		}
	}
	report.InputsAfter = len(g.Inputs)
	report.IRAfter = m.IRVersion
	return report, nil
}

// DropoutReport is what DropoutToIdentity did.
type DropoutReport struct {
	// Replaced is every Dropout node now an Identity, in graph order.
	Replaced []string `json:"replaced"`
	// Removed is the initializers and Constant nodes that only the Dropout
	// nodes read (their ratio and training_mode), removed with them.
	Removed []string `json:"removed"`
}

// DropoutToIdentity replaces every Dropout node of the top-level graph with
// an Identity of its data, which is what Dropout computes in inference.
//
// In inference Dropout returns its input unchanged at every opset: below 12
// it has no training mode at all, and from 12 on training_mode is false unless
// an input says otherwise. A node whose training_mode input is not the
// constant false, or whose mask output something reads, is refused with
// ErrUnsupported and the model is left unchanged. The Identity keeps the
// node's name and output, so nothing downstream is renamed; its ratio and
// training_mode constants are removed when nothing else reads them. Nodes in
// subgraphs are not rewritten.
func DropoutToIdentity(m *graph.Model) (DropoutReport, error) {
	if m == nil || m.Graph == nil {
		return DropoutReport{}, fmt.Errorf("%w: the model has no graph", ErrUnsupported)
	}
	g := m.Graph
	var targets []*graph.Node
	for _, n := range g.Nodes {
		if n.OpType != "Dropout" || !n.IsStandard() {
			continue
		}
		fail := func(format string, args ...any) error {
			return fmt.Errorf("%w: Dropout %q: %s", ErrUnsupported, n.Name, fmt.Sprintf(format, args...))
		}
		if n.Input(0) == "" || len(n.Outputs) == 0 || n.Outputs[0] == "" {
			return DropoutReport{}, fail("it has no data input or no output")
		}
		if len(n.Outputs) > 1 && n.Outputs[1] != "" && g.Used(n.Outputs[1]) {
			return DropoutReport{}, fail("its mask output %q is read, and an Identity has no mask", n.Outputs[1])
		}
		if mode := n.Input(2); mode != "" {
			t := g.Constant(mode)
			if t == nil {
				return DropoutReport{}, fail("its training_mode %q is computed at run time", mode)
			}
			if t.DataType != graph.Bool || t.Elements() != 1 || t.External {
				return DropoutReport{}, fail("its training_mode %q is not a bool scalar", mode)
			}
			on := false
			switch {
			case len(t.RawData) == 1:
				on = t.RawData[0] != 0
			case len(t.Int32Data) == 1:
				on = t.Int32Data[0] != 0
			default:
				return DropoutReport{}, fail("its training_mode %q cannot be read", mode)
			}
			if on {
				return DropoutReport{}, fail("its training_mode is true: it drops values, and an Identity does not")
			}
		}
		targets = append(targets, n)
	}

	var report DropoutReport
	constants := map[string]bool{}
	for _, n := range targets {
		constants[n.Input(1)] = true
		constants[n.Input(2)] = true
		n.OpType = "Identity"
		n.Inputs = []string{n.Inputs[0]}
		n.Outputs = []string{n.Outputs[0]}
		n.Attributes = nil
		report.Replaced = append(report.Replaced, n.Name)
	}
	if len(targets) > 0 && !g.HasSubgraphs() {
		report.Removed = removeUnusedConstants(g, constants)
	}
	if err := g.CheckTopological(); err != nil {
		return DropoutReport{}, fmt.Errorf("rewrite: the rewritten graph is out of order, which is a defect of the rewrite: %w", err)
	}
	return report, nil
}
