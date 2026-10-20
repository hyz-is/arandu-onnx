// Package onnxgraph reads the little of an ONNX model's protobuf encoding that
// calibration and the calibration-table check need, and rewrites one thing in
// it: the list of graph outputs.
//
// It is a hand-written reader of the protobuf wire format, on purpose. The
// fields it touches are a handful of numbers from onnx.proto, and a generated
// protobuf package would be a second dependency in every build that installs
// the module, for four fields.
//
// The field numbers are those of onnx.proto: ModelProto.graph = 7;
// GraphProto.node = 1, initializer = 5, input = 11, output = 12;
// NodeProto.input = 1, output = 2, name = 3, op_type = 4; TensorProto.name = 8;
// ValueInfoProto.name = 1.
package onnxgraph

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// ErrMalformed is a model whose bytes are not the protobuf encoding of an ONNX
// ModelProto this package can read.
var ErrMalformed = errors.New("onnxgraph: malformed model")

// The protobuf wire types.
const (
	wireVarint  = 0
	wireFixed64 = 1
	wireBytes   = 2
	wireFixed32 = 5
)

// The field numbers read here.
const (
	modelGraph = 7

	graphNode        = 1
	graphInitializer = 5
	graphInput       = 11
	graphOutput      = 12

	nodeInput  = 1
	nodeOutput = 2
	nodeName   = 3
	nodeOpType = 4

	tensorName    = 8
	valueInfoName = 1
)

// Node is one operator of the top-level graph.
type Node struct {
	// Name is the node's name, which may be empty.
	Name string
	// OpType is the operator, such as Conv.
	OpType string
	// Inputs and Outputs are the tensor names the node reads and writes. An
	// empty name is an optional input or output left out.
	Inputs  []string
	Outputs []string
}

// Graph is what this package reads of a model's top-level graph. Subgraphs of
// control-flow operators are not read.
type Graph struct {
	// Inputs is every graph input, in order, including those an initializer
	// also names.
	Inputs []string
	// Outputs is every graph output, in order.
	Outputs []string
	// Initializers is every constant tensor stored in the graph.
	Initializers []string
	// Nodes is every node, in the order the file lists them.
	Nodes []Node
}

// Activations returns the tensors whose values depend on the inputs: every
// graph input that is not an initializer, and every node output that is not
// a constant, each once and in the order the graph first names it. The
// constants are the initializers, the outputs of Constant nodes and the
// outputs of a node that reads only constants: such a tensor has the same
// value for every sample, so it has no range to calibrate, and its scalars
// (an exponent, an epsilon) are values ONNX Runtime's Go binding cannot
// return as outputs.
func (g Graph) Activations() []string {
	constant := make(map[string]bool, len(g.Initializers))
	for _, name := range g.Initializers {
		constant[name] = true
	}
	seen := map[string]bool{}
	out := []string{}
	add := func(name string) {
		if name == "" || constant[name] || seen[name] {
			return
		}
		seen[name] = true
		out = append(out, name)
	}
	for _, name := range g.Inputs {
		add(name)
	}
	for _, node := range g.Nodes {
		folded := node.OpType == "Constant"
		if !folded {
			reads := 0
			folded = true
			for _, name := range node.Inputs {
				if name == "" {
					continue
				}
				reads++
				if !constant[name] {
					folded = false
				}
			}
			folded = folded && reads > 0
		}
		for _, name := range node.Outputs {
			if folded && name != "" {
				constant[name] = true
			}
			add(name)
		}
	}
	return out
}

// Read parses a ModelProto and returns its top-level graph.
func Read(model []byte) (Graph, error) {
	graphBytes, err := graphField(model)
	if err != nil {
		return Graph{}, err
	}
	var g Graph
	err = fields(graphBytes, func(field, wire int, value []byte) error {
		if wire != wireBytes {
			return nil
		}
		switch field {
		case graphNode:
			node, err := readNode(value)
			if err != nil {
				return err
			}
			g.Nodes = append(g.Nodes, node)
		case graphInitializer:
			name, err := stringField(value, tensorName)
			if err != nil {
				return err
			}
			g.Initializers = append(g.Initializers, name)
		case graphInput:
			name, err := stringField(value, valueInfoName)
			if err != nil {
				return err
			}
			g.Inputs = append(g.Inputs, name)
		case graphOutput:
			name, err := stringField(value, valueInfoName)
			if err != nil {
				return err
			}
			g.Outputs = append(g.Outputs, name)
		}
		return nil
	})
	if err != nil {
		return Graph{}, err
	}
	return g, nil
}

// WithOutputs returns a copy of the model whose graph also lists the named
// tensors as outputs, after the ones it already has. A name the graph already
// outputs is not added twice.
//
// The outputs are written with a name and no type, which ONNX Runtime accepts
// and resolves by inferring the type from the node that produces the tensor.
// Every other byte of the model is kept as it was, in the same order.
func WithOutputs(model []byte, names []string) ([]byte, error) {
	g, err := Read(model)
	if err != nil {
		return nil, err
	}
	existing := make(map[string]bool, len(g.Outputs))
	for _, name := range g.Outputs {
		existing[name] = true
	}
	graphBytes, err := graphField(model)
	if err != nil {
		return nil, err
	}
	extended := append([]byte(nil), graphBytes...)
	for _, name := range names {
		if name == "" || existing[name] {
			continue
		}
		existing[name] = true
		valueInfo := appendBytes(nil, valueInfoName, []byte(name))
		extended = appendBytes(extended, graphOutput, valueInfo)
	}

	out := make([]byte, 0, len(model)+len(extended)-len(graphBytes)+8)
	err = rawFields(model, func(field, wire int, raw, value []byte) error {
		if field == modelGraph && wire == wireBytes {
			out = appendBytes(out, modelGraph, extended)
			return nil
		}
		out = append(out, raw...)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// graphField returns the bytes of ModelProto.graph, refusing a model with none
// or with more than one.
func graphField(model []byte) ([]byte, error) {
	var graph []byte
	count := 0
	err := fields(model, func(field, wire int, value []byte) error {
		if field == modelGraph {
			if wire != wireBytes {
				return fmt.Errorf("%w: ModelProto.graph has wire type %d", ErrMalformed, wire)
			}
			graph = value
			count++
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	switch count {
	case 0:
		return nil, fmt.Errorf("%w: the model has no graph", ErrMalformed)
	case 1:
		return graph, nil
	default:
		return nil, fmt.Errorf("%w: the model has %d graph fields, and a model merged from several is not read here", ErrMalformed, count)
	}
}

// readNode reads the fields of a NodeProto this package uses.
func readNode(data []byte) (Node, error) {
	var node Node
	err := fields(data, func(field, wire int, value []byte) error {
		if wire != wireBytes {
			return nil
		}
		switch field {
		case nodeInput:
			node.Inputs = append(node.Inputs, string(value))
		case nodeOutput:
			node.Outputs = append(node.Outputs, string(value))
		case nodeName:
			node.Name = string(value)
		case nodeOpType:
			node.OpType = string(value)
		}
		return nil
	})
	return node, err
}

// stringField returns the last value of a length-delimited field, which is
// what protobuf makes of a scalar field written more than once.
func stringField(data []byte, number int) (string, error) {
	var out string
	err := fields(data, func(field, wire int, value []byte) error {
		if field == number && wire == wireBytes {
			out = string(value)
		}
		return nil
	})
	return out, err
}

// fields calls fn with every field of a message, and the payload of each.
// For a varint or a fixed-width field the payload is its raw bytes.
func fields(data []byte, fn func(field, wire int, value []byte) error) error {
	return rawFields(data, func(field, wire int, _, value []byte) error {
		return fn(field, wire, value)
	})
}

// rawFields is fields with the whole encoded field, tag included, as well.
func rawFields(data []byte, fn func(field, wire int, raw, value []byte) error) error {
	for offset := 0; offset < len(data); {
		start := offset
		tag, n := binary.Uvarint(data[offset:])
		if n <= 0 {
			return fmt.Errorf("%w: unreadable tag at byte %d", ErrMalformed, offset)
		}
		offset += n
		field, wire := int(tag>>3), int(tag&7)
		if field == 0 {
			return fmt.Errorf("%w: field number 0 at byte %d", ErrMalformed, start)
		}
		var value []byte
		switch wire {
		case wireVarint:
			_, n := binary.Uvarint(data[offset:])
			if n <= 0 {
				return fmt.Errorf("%w: unreadable varint at byte %d", ErrMalformed, offset)
			}
			value = data[offset : offset+n]
			offset += n
		case wireFixed64:
			if len(data)-offset < 8 {
				return fmt.Errorf("%w: truncated fixed64 at byte %d", ErrMalformed, offset)
			}
			value = data[offset : offset+8]
			offset += 8
		case wireFixed32:
			if len(data)-offset < 4 {
				return fmt.Errorf("%w: truncated fixed32 at byte %d", ErrMalformed, offset)
			}
			value = data[offset : offset+4]
			offset += 4
		case wireBytes:
			length, n := binary.Uvarint(data[offset:])
			if n <= 0 {
				return fmt.Errorf("%w: unreadable length at byte %d", ErrMalformed, offset)
			}
			offset += n
			if length > uint64(len(data)-offset) {
				return fmt.Errorf("%w: a field of %d bytes at byte %d runs past the end", ErrMalformed, length, offset)
			}
			value = data[offset : offset+int(length)]
			offset += int(length)
		default:
			return fmt.Errorf("%w: wire type %d at byte %d is not one an ONNX model uses", ErrMalformed, wire, start)
		}
		if err := fn(field, wire, data[start:offset], value); err != nil {
			return err
		}
	}
	return nil
}

// appendBytes appends a length-delimited field.
func appendBytes(out []byte, field int, value []byte) []byte {
	out = binary.AppendUvarint(out, uint64(field)<<3|wireBytes)
	out = binary.AppendUvarint(out, uint64(len(value)))
	return append(out, value...)
}
