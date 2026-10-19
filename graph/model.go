// Package graph reads, edits and writes ONNX models in pure Go.
//
// A model is decoded from the protobuf encoding of an ONNX ModelProto into
// Model, Graph, Node, Attribute, Tensor and ValueInfo values that a rewrite can
// change, and Marshal encodes it again. Only the fields a rewrite needs are
// decoded; every other field of those messages -- doc strings, metadata,
// training information, local functions, sparse initializers, quantization
// annotations, subgraphs, the external-data references of a tensor -- is kept
// as the bytes it was read as and written back unchanged. A model loaded and
// marshalled with nothing changed is the same bytes again, when it was
// written as the C++ protobuf library writes it: fields in ascending number,
// which is what every ONNX exporter produces.
//
// The field numbers are those of onnx.proto (MIT, the ONNX project). The
// package reads no file and calls no runtime: it works on bytes, and loading
// a model into ONNX Runtime stays the job of onnx.Runtime and its manifests.
package graph

import (
	"errors"
	"fmt"
)

// ErrMalformed is wrapped by every error that reports bytes which are not the
// protobuf encoding of an ONNX model this package can read.
var ErrMalformed = errors.New("graph: malformed model")

// The field numbers of onnx.proto this package decodes.
const (
	modelIRVersion = 1
	modelGraph     = 7
	modelOpset     = 8

	opsetDomain  = 1
	opsetVersion = 2

	graphNode        = 1
	graphName        = 2
	graphInitializer = 5
	graphInput       = 11
	graphOutput      = 12
	graphValueInfo   = 13

	nodeInput     = 1
	nodeOutput    = 2
	nodeName      = 3
	nodeOpType    = 4
	nodeAttribute = 5
	nodeDomain    = 7

	attrName    = 1
	attrF       = 2
	attrI       = 3
	attrS       = 4
	attrT       = 5
	attrG       = 6
	attrFloats  = 7
	attrInts    = 8
	attrStrings = 9
	attrGraphs  = 11
	attrType    = 20

	tensorDims         = 1
	tensorDataType     = 2
	tensorFloatData    = 4
	tensorInt32Data    = 5
	tensorInt64Data    = 7
	tensorName         = 8
	tensorRawData      = 9
	tensorDataLocation = 14

	valueInfoName = 1
	valueInfoType = 2
)

// DefaultDomain is the domain of the standard ONNX operators, which a model
// may also spell "ai.onnx".
const DefaultDomain = ""

// isDefaultDomain reports whether a domain names the standard operators.
func isDefaultDomain(domain string) bool { return domain == "" || domain == "ai.onnx" }

// presence records which scalar fields of a message were present when it was
// read, by field number, so that a field written explicitly with its zero
// value is written again.
type presence uint64

func (p *presence) mark(num int) {
	if num < 64 {
		*p |= 1 << uint(num)
	}
}

func (p presence) has(num int) bool { return num < 64 && p&(1<<uint(num)) != 0 }

// Model is an ONNX ModelProto.
type Model struct {
	// IRVersion is the ONNX IR version the model is written in.
	IRVersion int64
	// OpsetImports is the operator set version of each domain the model
	// uses. The standard operators are the domain "" (or "ai.onnx").
	OpsetImports []*OpsetImport
	// Graph is the model's graph.
	Graph *Graph

	set   presence
	extra []rawField
}

// OpsetImport is an ONNX OperatorSetIdProto.
type OpsetImport struct {
	// Domain is the operator domain; "" is the standard operators.
	Domain string
	// Version is the operator set version.
	Version int64

	set   presence
	extra []rawField
}

// Graph is an ONNX GraphProto.
//
// Nodes are in the order the file lists them, which ONNX requires to be a
// topological order; a rewrite that inserts a node keeps it so.
type Graph struct {
	// Name is the graph's name.
	Name string
	// Nodes is every node of the graph.
	Nodes []*Node
	// Initializers is every constant tensor stored in the graph.
	Initializers []*Tensor
	// Inputs is every graph input, including, in models written before IR
	// version 4, the initializers.
	Inputs []*ValueInfo
	// Outputs is every graph output.
	Outputs []*ValueInfo
	// ValueInfo is the type and shape the file records for intermediate
	// tensors. It is often partial or empty.
	ValueInfo []*ValueInfo

	set   presence
	extra []rawField
}

// Node is an ONNX NodeProto.
type Node struct {
	// Inputs is the tensor names the node reads. An empty name is an
	// optional input left out.
	Inputs []string
	// Outputs is the tensor names the node writes.
	Outputs []string
	// Name is the node's name, which may be empty.
	Name string
	// OpType is the operator, such as Conv.
	OpType string
	// Domain is the operator's domain; "" is the standard operators.
	Domain string
	// Attributes is the node's attributes, in order.
	Attributes []*Attribute

	set   presence
	extra []rawField
}

// AttributeType is the type of an attribute, AttributeProto.AttributeType.
type AttributeType int32

// The attribute types this package reads and writes. The others (graphs,
// sparse tensors, type protos, lists of tensors) are kept as they were read.
const (
	AttributeFloat   AttributeType = 1
	AttributeInt     AttributeType = 2
	AttributeString  AttributeType = 3
	AttributeTensor  AttributeType = 4
	AttributeGraph   AttributeType = 5
	AttributeFloats  AttributeType = 6
	AttributeInts    AttributeType = 7
	AttributeStrings AttributeType = 8
	AttributeGraphs  AttributeType = 10
)

// Attribute is an ONNX AttributeProto.
type Attribute struct {
	// Name is the attribute's name.
	Name string
	// Type is which of the value fields holds its value.
	Type AttributeType
	// F is the value of a float attribute.
	F float32
	// I is the value of an int attribute.
	I int64
	// S is the value of a string attribute.
	S []byte
	// T is the value of a tensor attribute.
	T *Tensor
	// Floats is the value of a floats attribute.
	Floats []float32
	// Ints is the value of an ints attribute.
	Ints []int64
	// Strings is the value of a strings attribute.
	Strings [][]byte

	subgraph bool
	set      presence
	extra    []rawField
}

// Tensor is an ONNX TensorProto: an initializer, or the value of a Constant
// node or a tensor attribute.
type Tensor struct {
	// Dims is the tensor's shape; empty for a scalar.
	Dims []int64
	// DataType is the element type.
	DataType DataType
	// Name is the tensor's name.
	Name string
	// RawData is the elements in little-endian binary, when the file stores
	// them so.
	RawData []byte
	// FloatData is the elements of a float tensor stored as a list.
	FloatData []float32
	// Int32Data is the elements of an int32, int16, int8, uint16, uint8,
	// bool or float16 tensor stored as a list.
	Int32Data []int32
	// Int64Data is the elements of an int64 tensor stored as a list.
	Int64Data []int64
	// External reports a tensor whose elements are in an external data file;
	// its location entries are kept as they were read, and its values cannot
	// be read here.
	External bool

	set   presence
	extra []rawField
}

// ValueInfo is an ONNX ValueInfoProto: the name and, when the file records
// it, the type and shape of a tensor.
type ValueInfo struct {
	// Name is the tensor's name.
	Name string

	typ   []byte
	set   presence
	extra []rawField
}

// Load decodes a model.
//
// The returned model shares memory with data: data must not be changed while
// the model is in use.
func Load(data []byte) (*Model, error) {
	m := &Model{}
	graphs := 0
	err := each(data, func(f field) error {
		switch f.num {
		case modelIRVersion:
			v, err := f.int()
			m.IRVersion = v
			m.set.mark(f.num)
			return err
		case modelGraph:
			b, err := f.bytes()
			if err != nil {
				return err
			}
			g, err := decodeGraph(b)
			if err != nil {
				return err
			}
			m.Graph = g
			graphs++
		case modelOpset:
			b, err := f.bytes()
			if err != nil {
				return err
			}
			o, err := decodeOpset(b)
			if err != nil {
				return err
			}
			m.OpsetImports = append(m.OpsetImports, o)
		default:
			m.extra = append(m.extra, f.keep())
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	switch graphs {
	case 0:
		return nil, fmt.Errorf("%w: the model has no graph", ErrMalformed)
	case 1:
		return m, nil
	default:
		return nil, fmt.Errorf("%w: the model has %d graph fields, and a model merged from several is not read here", ErrMalformed, graphs)
	}
}

// Marshal encodes the model.
func (m *Model) Marshal() ([]byte, error) {
	if m.Graph == nil {
		return nil, fmt.Errorf("%w: the model has no graph", ErrMalformed)
	}
	var e encoder
	if m.set.has(modelIRVersion) || m.IRVersion != 0 {
		e.varint(modelIRVersion, uint64(m.IRVersion))
	}
	e.bytes(modelGraph, m.Graph.encode())
	for _, o := range m.OpsetImports {
		e.bytes(modelOpset, o.encode())
	}
	e.keep(m.extra)
	return e.encode(), nil
}

// Opset returns the operator set version the model imports for a domain,
// with "" and "ai.onnx" the same domain.
func (m *Model) Opset(domain string) (int64, bool) {
	for _, o := range m.OpsetImports {
		if o.Domain == domain || (isDefaultDomain(domain) && isDefaultDomain(o.Domain)) {
			return o.Version, true
		}
	}
	return 0, false
}

// SetOpset sets the operator set version of a domain, adding the import when
// the model has none.
func (m *Model) SetOpset(domain string, version int64) {
	for _, o := range m.OpsetImports {
		if o.Domain == domain || (isDefaultDomain(domain) && isDefaultDomain(o.Domain)) {
			o.Version = version
			return
		}
	}
	m.OpsetImports = append(m.OpsetImports, &OpsetImport{Domain: domain, Version: version})
}

// HasFunctions reports whether the model carries local functions, whose
// bodies this package keeps unread.
func (m *Model) HasFunctions() bool {
	for _, f := range m.extra {
		if f.num == 25 {
			return true
		}
	}
	return false
}

func decodeOpset(data []byte) (*OpsetImport, error) {
	o := &OpsetImport{}
	err := each(data, func(f field) error {
		var err error
		switch f.num {
		case opsetDomain:
			o.Domain, err = f.str()
			o.set.mark(f.num)
		case opsetVersion:
			o.Version, err = f.int()
			o.set.mark(f.num)
		default:
			o.extra = append(o.extra, f.keep())
		}
		return err
	})
	return o, err
}

func (o *OpsetImport) encode() []byte {
	var e encoder
	if o.set.has(opsetDomain) || o.Domain != "" {
		e.str(opsetDomain, o.Domain)
	}
	if o.set.has(opsetVersion) || o.Version != 0 {
		e.varint(opsetVersion, uint64(o.Version))
	}
	e.keep(o.extra)
	return e.encode()
}

func decodeGraph(data []byte) (*Graph, error) {
	g := &Graph{}
	err := each(data, func(f field) error {
		switch f.num {
		case graphName:
			s, err := f.str()
			g.Name = s
			g.set.mark(f.num)
			return err
		case graphNode, graphInitializer, graphInput, graphOutput, graphValueInfo:
		default:
			g.extra = append(g.extra, f.keep())
			return nil
		}
		b, err := f.bytes()
		if err != nil {
			return err
		}
		switch f.num {
		case graphNode:
			n, err := decodeNode(b)
			if err != nil {
				return err
			}
			g.Nodes = append(g.Nodes, n)
		case graphInitializer:
			t, err := decodeTensor(b)
			if err != nil {
				return err
			}
			g.Initializers = append(g.Initializers, t)
		default:
			v, err := decodeValueInfo(b)
			if err != nil {
				return err
			}
			switch f.num {
			case graphInput:
				g.Inputs = append(g.Inputs, v)
			case graphOutput:
				g.Outputs = append(g.Outputs, v)
			default:
				g.ValueInfo = append(g.ValueInfo, v)
			}
		}
		return nil
	})
	return g, err
}

func (g *Graph) encode() []byte {
	var e encoder
	for _, n := range g.Nodes {
		e.bytes(graphNode, n.encode())
	}
	if g.set.has(graphName) || g.Name != "" {
		e.str(graphName, g.Name)
	}
	for _, t := range g.Initializers {
		e.bytes(graphInitializer, t.encode())
	}
	for _, v := range g.Inputs {
		e.bytes(graphInput, v.encode())
	}
	for _, v := range g.Outputs {
		e.bytes(graphOutput, v.encode())
	}
	for _, v := range g.ValueInfo {
		e.bytes(graphValueInfo, v.encode())
	}
	e.keep(g.extra)
	return e.encode()
}

func decodeNode(data []byte) (*Node, error) {
	n := &Node{}
	err := each(data, func(f field) error {
		var err error
		var s string
		switch f.num {
		case nodeInput:
			s, err = f.str()
			n.Inputs = append(n.Inputs, s)
		case nodeOutput:
			s, err = f.str()
			n.Outputs = append(n.Outputs, s)
		case nodeName:
			n.Name, err = f.str()
			n.set.mark(f.num)
		case nodeOpType:
			n.OpType, err = f.str()
			n.set.mark(f.num)
		case nodeDomain:
			n.Domain, err = f.str()
			n.set.mark(f.num)
		case nodeAttribute:
			var b []byte
			if b, err = f.bytes(); err != nil {
				return err
			}
			var a *Attribute
			if a, err = decodeAttribute(b); err != nil {
				return err
			}
			n.Attributes = append(n.Attributes, a)
		default:
			n.extra = append(n.extra, f.keep())
		}
		return err
	})
	return n, err
}

func (n *Node) encode() []byte {
	var e encoder
	for _, s := range n.Inputs {
		e.str(nodeInput, s)
	}
	for _, s := range n.Outputs {
		e.str(nodeOutput, s)
	}
	if n.set.has(nodeName) || n.Name != "" {
		e.str(nodeName, n.Name)
	}
	if n.set.has(nodeOpType) || n.OpType != "" {
		e.str(nodeOpType, n.OpType)
	}
	for _, a := range n.Attributes {
		e.bytes(nodeAttribute, a.encode())
	}
	if n.set.has(nodeDomain) || n.Domain != "" {
		e.str(nodeDomain, n.Domain)
	}
	e.keep(n.extra)
	return e.encode()
}

func decodeAttribute(data []byte) (*Attribute, error) {
	a := &Attribute{}
	err := each(data, func(f field) error {
		var err error
		switch f.num {
		case attrName:
			a.Name, err = f.str()
			a.set.mark(f.num)
		case attrType:
			var v int64
			v, err = f.int()
			a.Type = AttributeType(v)
			a.set.mark(f.num)
		case attrF:
			a.F, err = f.float()
			a.set.mark(f.num)
		case attrI:
			a.I, err = f.int()
			a.set.mark(f.num)
		case attrS:
			a.S, err = f.bytes()
			a.set.mark(f.num)
		case attrT:
			var b []byte
			if b, err = f.bytes(); err != nil {
				return err
			}
			a.T, err = decodeTensor(b)
		case attrFloats:
			a.Floats, err = f.appendFloats(a.Floats)
		case attrInts:
			a.Ints, err = f.appendInts(a.Ints)
		case attrStrings:
			var b []byte
			b, err = f.bytes()
			a.Strings = append(a.Strings, b)
		default:
			if f.num == attrG || f.num == attrGraphs {
				a.subgraph = true
			}
			a.extra = append(a.extra, f.keep())
		}
		return err
	})
	return a, err
}

func (a *Attribute) encode() []byte {
	var e encoder
	if a.set.has(attrName) || a.Name != "" {
		e.str(attrName, a.Name)
	}
	if a.set.has(attrF) || a.F != 0 || a.Type == AttributeFloat {
		e.float(attrF, a.F)
	}
	if a.set.has(attrI) || a.I != 0 || a.Type == AttributeInt {
		e.varint(attrI, uint64(a.I))
	}
	if a.set.has(attrS) || len(a.S) != 0 || a.Type == AttributeString {
		e.bytes(attrS, a.S)
	}
	if a.T != nil {
		e.bytes(attrT, a.T.encode())
	}
	e.floats(attrFloats, a.Floats)
	e.ints(attrInts, a.Ints)
	for _, s := range a.Strings {
		e.bytes(attrStrings, s)
	}
	if a.set.has(attrType) || a.Type != 0 {
		e.varint(attrType, uint64(a.Type))
	}
	e.keep(a.extra)
	return e.encode()
}

func decodeTensor(data []byte) (*Tensor, error) {
	t := &Tensor{}
	err := each(data, func(f field) error {
		var err error
		switch f.num {
		case tensorDims:
			t.Dims, err = f.appendInts(t.Dims)
		case tensorDataType:
			var v int64
			v, err = f.int()
			t.DataType = DataType(v)
			t.set.mark(f.num)
		case tensorName:
			t.Name, err = f.str()
			t.set.mark(f.num)
		case tensorRawData:
			t.RawData, err = f.bytes()
			t.set.mark(f.num)
		case tensorFloatData:
			t.FloatData, err = f.appendFloats(t.FloatData)
		case tensorInt32Data:
			var vs []int64
			vs, err = f.appendInts(nil)
			for _, v := range vs {
				t.Int32Data = append(t.Int32Data, int32(v))
			}
		case tensorInt64Data:
			t.Int64Data, err = f.appendInts(t.Int64Data)
		case tensorDataLocation:
			var v int64
			v, err = f.int()
			t.External = v == 1
			t.set.mark(f.num)
		default:
			t.extra = append(t.extra, f.keep())
		}
		return err
	})
	return t, err
}

func (t *Tensor) encode() []byte {
	var e encoder
	e.ints(tensorDims, t.Dims)
	if t.set.has(tensorDataType) || t.DataType != 0 {
		e.varint(tensorDataType, uint64(t.DataType))
	}
	e.packedFloats(tensorFloatData, t.FloatData)
	if len(t.Int32Data) > 0 {
		vs := make([]int64, len(t.Int32Data))
		for i, v := range t.Int32Data {
			vs[i] = int64(v)
		}
		e.packedInts(tensorInt32Data, vs)
	}
	e.packedInts(tensorInt64Data, t.Int64Data)
	if t.set.has(tensorName) || t.Name != "" {
		e.str(tensorName, t.Name)
	}
	if t.set.has(tensorRawData) || len(t.RawData) != 0 {
		e.bytes(tensorRawData, t.RawData)
	}
	if t.set.has(tensorDataLocation) || t.External {
		location := uint64(0)
		if t.External {
			location = 1
		}
		e.varint(tensorDataLocation, location)
	}
	e.keep(t.extra)
	return e.encode()
}

func decodeValueInfo(data []byte) (*ValueInfo, error) {
	v := &ValueInfo{}
	err := each(data, func(f field) error {
		var err error
		switch f.num {
		case valueInfoName:
			v.Name, err = f.str()
			v.set.mark(f.num)
		case valueInfoType:
			v.typ, err = f.bytes()
			v.set.mark(f.num)
		default:
			v.extra = append(v.extra, f.keep())
		}
		return err
	})
	return v, err
}

func (v *ValueInfo) encode() []byte {
	var e encoder
	if v.set.has(valueInfoName) || v.Name != "" {
		e.str(valueInfoName, v.Name)
	}
	if v.set.has(valueInfoType) || len(v.typ) != 0 {
		e.bytes(valueInfoType, v.typ)
	}
	e.keep(v.extra)
	return e.encode()
}
