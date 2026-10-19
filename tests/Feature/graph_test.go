package feature_test

import (
	"bytes"
	"errors"
	"slices"
	"testing"

	"github.com/hyz-is/arandu-onnx/graph"
)

func TestAModelLoadedAndMarshalledUnchangedIsTheSameBytes(t *testing.T) {
	t.Parallel()

	// A doc string on the model (field 6), metadata (14), a node doc string
	// (6) and a value info doc string (3) are fields the package does not
	// interpret; they have to come back where they were.
	metadata := message{}.text(1, "author").text(2, "tests")
	documented := message{}.text(1, "x").text(2, "unused").text(3, "Mul").text(4, "Mul").text(6, "a node doc")
	g := message{}.
		sub(1, documented).
		text(2, "documented").
		sub(11, valueInfo("x", protoFloat, -1, 3).text(3, "an input doc")).
		sub(12, valueInfo("unused", protoFloat, -1, 3))
	withExtras := message{}.integer(1, 8).text(2, "arandu-onnx-tests").text(6, "a model doc").
		sub(7, g).sub(8, message{}.text(1, "").integer(2, 17)).sub(14, metadata)

	for name, body := range map[string][]byte{
		"affine":      affineModel(),
		"cast":        castModel(),
		"reshape":     reshapeModel(),
		"with extras": withExtras,
	} {
		m, err := graph.Load(body)
		if err != nil {
			t.Fatalf("%s: Load: %v", name, err)
		}
		out, err := m.Marshal()
		if err != nil {
			t.Fatalf("%s: Marshal: %v", name, err)
		}
		if !bytes.Equal(out, body) {
			t.Errorf("%s: %d bytes in, %d bytes out, and they differ", name, len(body), len(out))
		}
	}
}

func TestTheGraphIsReadAsItsFileWritesIt(t *testing.T) {
	t.Parallel()

	m, err := graph.Load(affineModel())
	if err != nil {
		t.Fatal(err)
	}
	if m.IRVersion != 7 {
		t.Errorf("IRVersion = %d, want 7", m.IRVersion)
	}
	if v, ok := m.Opset(""); !ok || v != 13 {
		t.Errorf("Opset = %d, %v; want 13", v, ok)
	}
	if v, ok := m.Opset("ai.onnx"); !ok || v != 13 {
		t.Errorf(`Opset("ai.onnx") = %d, %v; "" and "ai.onnx" are one domain`, v, ok)
	}
	g := m.Graph
	if g.Name != "affine" || len(g.Nodes) != 3 || len(g.Initializers) != 2 {
		t.Fatalf("graph = %q with %d nodes and %d initializers", g.Name, len(g.Nodes), len(g.Initializers))
	}
	argmax := g.Nodes[2]
	if argmax.OpType != "ArgMax" || argmax.Int("axis", -1) != 1 || argmax.Int("keepdims", -1) != 0 || argmax.Int("absent", 42) != 42 {
		t.Errorf("ArgMax attributes = %+v", argmax.Attributes)
	}
	w, err := g.Initializer("w").Float32s()
	if err != nil || !slices.Equal(w, []float32{1, 2, 3, 4, 5, 6}) {
		t.Errorf("w = %v, %v", w, err)
	}
	if _, err := g.Initializer("w").Int64s(); !errors.Is(err, graph.ErrUnreadable) {
		t.Errorf("Int64s of a float tensor = %v, want ErrUnreadable", err)
	}
	if p := g.Producer("xw"); p == nil || p.OpType != "MatMul" {
		t.Errorf("Producer(xw) = %+v", p)
	}
	if c := g.Consumers("y"); len(c) != 1 || c[0].OpType != "ArgMax" {
		t.Errorf("Consumers(y) = %+v", c)
	}
	x, ok := g.TypeOf("x")
	if !ok || x.ElemType != graph.Float || len(x.Shape) != 2 || x.Shape[0].Value != 0 || x.Shape[0].Param == "" || x.Shape[1].Value != 3 {
		t.Errorf("TypeOf(x) = %+v, %v; want float [dynamic, 3]", x, ok)
	}
	if _, static := x.Static(); static {
		t.Error("x has a dynamic batch and Static says it is fixed")
	}
	if wt, ok := g.TypeOf("w"); !ok || len(wt.Shape) != 2 || wt.Shape[0].Value != 3 {
		t.Errorf("TypeOf(w) from the initializer = %+v, %v", wt, ok)
	}
	if g.Constant("w") == nil || g.Constant("x") != nil || g.Constant("xw") != nil {
		t.Error("Constant: w is a constant, x an input and xw computed")
	}
	if err := g.CheckTopological(); err != nil {
		t.Errorf("CheckTopological: %v", err)
	}
	g.Nodes[0], g.Nodes[1] = g.Nodes[1], g.Nodes[0]
	if err := g.CheckTopological(); !errors.Is(err, graph.ErrMalformed) {
		t.Errorf("CheckTopological of swapped nodes = %v, want ErrMalformed", err)
	}
}

func TestConstantNodesNamerAndNewTensorsRoundTrip(t *testing.T) {
	t.Parallel()

	m, err := graph.Load(reshapeModel())
	if err != nil {
		t.Fatal(err)
	}
	g := m.Graph
	image := g.Constant("shape")
	if image == nil {
		t.Fatal("the Constant node's value is not read")
	}
	if v, err := image.Int64s(); err != nil || !slices.Equal(v, []int64{4, 5}) {
		t.Errorf("shape = %v, %v", v, err)
	}

	namer := graph.NewNamer(g)
	if got := namer.Fresh("x"); got != "x_1" {
		t.Errorf("Fresh(x) = %s, want x_1: x is taken", got)
	}
	if got := namer.Fresh("x"); got != "x_2" {
		t.Errorf("Fresh(x) again = %s, want x_2", got)
	}
	if got := namer.Fresh("new"); got != "new" {
		t.Errorf("Fresh(new) = %s", got)
	}

	g.Initializers = append(g.Initializers,
		graph.Int8Tensor("q", []int64{3}, []int8{-127, 0, 127}),
		graph.Float32Tensor("s", nil, []float32{0.5}))
	g.ValueInfo = append(g.ValueInfo, graph.NewValueInfo("v", graph.TensorType{
		ElemType: graph.Int8, Ranked: true, Shape: []graph.Dim{{Value: 2}, {Param: "n"}},
	}))
	m.SetOpset("", 19)
	m.SetOpset("com.example", 1)
	body, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	back, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if q, err := back.Graph.Initializer("q").Int8s(); err != nil || !slices.Equal(q, []int8{-127, 0, 127}) {
		t.Errorf("q = %v, %v", q, err)
	}
	if s, err := back.Graph.Initializer("s").Float32s(); err != nil || len(s) != 1 || s[0] != 0.5 {
		t.Errorf("s = %v, %v", s, err)
	}
	v, ok := back.Graph.TypeOf("v")
	if !ok || v.ElemType != graph.Int8 || v.Shape[0].Value != 2 || v.Shape[1].Param != "n" {
		t.Errorf("v = %+v", v)
	}
	if o, _ := back.Opset(""); o != 19 {
		t.Errorf("opset = %d, want 19", o)
	}
	if o, ok := back.Opset("com.example"); !ok || o != 1 {
		t.Errorf("com.example opset = %d, %v", o, ok)
	}
}

func TestBytesThatAreNotAModelAreRefused(t *testing.T) {
	t.Parallel()

	twoGraphs := message{}.sub(7, message{}).sub(7, message{})
	for name, body := range map[string][]byte{
		"empty":      {},
		"truncated":  affineModel()[:20],
		"no graph":   message{}.integer(1, 7),
		"two graphs": twoGraphs,
		"bad wire":   {0x0b},
	} {
		if _, err := graph.Load(body); !errors.Is(err, graph.ErrMalformed) {
			t.Errorf("%s: Load = %v, want ErrMalformed", name, err)
		}
	}
}

// reshapeModel is y = Reshape(x, shape) with shape {4, 5} from a Constant
// node and an unused int64 initializer, at opset 18.
func reshapeModel() []byte {
	g := message{}.
		sub(1, node("Constant", nil, []string{"shape"}, intsAttribute("value_ints", 4, 5))).
		sub(1, node("Reshape", []string{"x", "shape"}, []string{"y"})).
		text(2, "reshape").
		sub(5, int64Initializer("spare", []int64{7, 8, 9}, 3)).
		sub(11, valueInfo("x", protoFloat, 20)).
		sub(12, valueInfo("y", protoFloat, 4, 5))
	return modelAt(g, 8, 18)
}
