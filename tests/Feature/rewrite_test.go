package feature_test

import (
	"bytes"
	"errors"
	"slices"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/rewrite"
)

// col2imCase is a one-node Col2Im graph and the input it is run on.
type col2imCase struct {
	name                    string
	channels                int64
	image, block            []int64
	strides, pads, dilation []int64
	batch                   int64
	// blocks is the number of sliding blocks per dimension, worked out by
	// hand: floor((image + pad_begin + pad_end - dilation*(block-1) - 1) /
	// stride) + 1.
	blocks []int64
	// outputPadding is the remainder of that division, by hand.
	outputPadding []int64
	// constantNodes writes image_shape and block_shape as Constant nodes
	// instead of initializers.
	constantNodes bool
	// unrecorded leaves dimension 1 of the input out of the graph's value
	// info.
	unrecorded bool
}

func (c col2imCase) k() int64 {
	k := int64(1)
	for _, b := range c.block {
		k *= b
	}
	return k
}

func (c col2imCase) l() int64 {
	l := int64(1)
	for _, b := range c.blocks {
		l *= b
	}
	return l
}

// model writes the case as a Col2Im graph at opset 18.
func (c col2imCase) model() []byte {
	var attributes []message
	if c.strides != nil {
		attributes = append(attributes, intsAttribute("strides", c.strides...))
	}
	if c.pads != nil {
		attributes = append(attributes, intsAttribute("pads", c.pads...))
	}
	if c.dilation != nil {
		attributes = append(attributes, intsAttribute("dilations", c.dilation...))
	}
	rank := int64(len(c.image))
	g := message{}
	if c.constantNodes {
		g = g.
			sub(1, node("Constant", nil, []string{"image_shape"}, intsAttribute("value_ints", c.image...))).
			sub(1, node("Constant", nil, []string{"block_shape"}, intsAttribute("value_ints", c.block...)))
	}
	g = g.sub(1, node("Col2Im", []string{"x", "image_shape", "block_shape"}, []string{"y"}, attributes...)).
		text(2, "col2im")
	if !c.constantNodes {
		g = g.sub(5, int64Initializer("image_shape", c.image, rank)).
			sub(5, int64Initializer("block_shape", c.block, rank))
	}
	channels := c.channels * c.k()
	if c.unrecorded {
		channels = -1
	}
	g = g.sub(11, valueInfo("x", protoFloat, c.batch, channels, c.l())).
		sub(12, valueInfo("y", protoFloat, append([]int64{c.batch, c.channels}, c.image...)...))
	return modelAt(g, 8, 18)
}

func (c col2imCase) input(seed uint64) onnx.Tensor {
	return randomTensor(seed, c.batch, c.channels*c.k(), c.l())
}

var col2imCases = []col2imCase{
	{
		// MiVOLO's shape, scaled down: k 3, stride 2, pads 1, and an image
		// whose last row and column no block reaches.
		name: "mivolo-like", channels: 3, image: []int64{8, 8}, block: []int64{3, 3},
		strides: []int64{2, 2}, pads: []int64{1, 1, 1, 1}, dilation: []int64{1, 1}, batch: 1,
		// (8 + 2 - 2 - 1) / 2 = 3 rest 1.
		blocks: []int64{4, 4}, outputPadding: []int64{1, 1},
	},
	{
		// Every attribute away from its default, and asymmetric.
		name: "asymmetric", channels: 2, image: []int64{5, 7}, block: []int64{2, 3},
		strides: []int64{1, 2}, pads: []int64{1, 0, 0, 1}, dilation: []int64{2, 1}, batch: 2,
		// rows: 5 + 1 + 0 - 2 - 1 = 3, / 1 = 3 rest 0; columns: 7 + 0 + 1 - 2 - 1 = 5, / 2 = 2 rest 1.
		blocks: []int64{4, 3}, outputPadding: []int64{0, 1},
	},
	{
		// Defaults only, from Constant nodes, with the input's channels not
		// recorded in the graph.
		name: "defaults", channels: 4, image: []int64{4, 5}, block: []int64{2, 2}, batch: 3,
		// rows: 4 - 1 - 1 = 2, + 1 = 3; columns: 5 - 1 - 1 = 3, + 1 = 4.
		blocks: []int64{3, 4}, outputPadding: []int64{0, 0},
		constantNodes: true, unrecorded: true,
	},
}

func TestCol2ImIsRewrittenIntoAGroupedConvTransposeThatComputesTheSame(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})

	for _, c := range col2imCases {
		t.Run(c.name, func(t *testing.T) {
			original := c.model()
			m, err := graph.Load(original)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			opts := rewrite.Col2ImOptions{}
			if c.unrecorded {
				if _, err := rewrite.Col2ImToConvTranspose(m, opts); !errors.Is(err, rewrite.ErrUnsupported) {
					t.Fatalf("without the channels: %v, want ErrUnsupported", err)
				}
				opts.InputChannels = map[string]int64{"x": c.channels * c.k()}
			}
			report, err := rewrite.Col2ImToConvTranspose(m, opts)
			if err != nil {
				t.Fatalf("Col2ImToConvTranspose: %v", err)
			}
			rewritten, err := m.Marshal()
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}

			if len(report.Rewritten) != 1 {
				t.Fatalf("Rewritten = %+v, want one node", report.Rewritten)
			}
			r := report.Rewritten[0]
			if r.Channels != c.channels || !slices.Equal(r.Blocks, c.blocks) || !slices.Equal(r.OutputPadding, c.outputPadding) {
				t.Errorf("rewrite = %+v, want %d channels, blocks %v, output padding %v", r, c.channels, c.blocks, c.outputPadding)
			}
			wantRemoved := "block_shape,image_shape"
			if got := strings.Join(report.Removed, ","); got != wantRemoved {
				t.Errorf("Removed = %s, want %s", got, wantRemoved)
			}

			// The rewritten file, read back, has no Col2Im and the one-hot
			// grouped ConvTranspose.
			back, err := graph.Load(rewritten)
			if err != nil {
				t.Fatalf("Load(rewritten): %v", err)
			}
			counts := back.Graph.OpCounts()
			if counts["Col2Im"] != 0 || counts["ConvTranspose"] != 1 || counts["Reshape"] != 1 || counts["Constant"] != 0 {
				t.Errorf("ops after = %v", back.Graph.SortedOpCounts())
			}
			if report.NodesAfter != len(back.Graph.Nodes) {
				t.Errorf("NodesAfter = %d, the graph has %d", report.NodesAfter, len(back.Graph.Nodes))
			}
			var deconv *graph.Node
			for _, n := range back.Graph.Nodes {
				if n.OpType == "ConvTranspose" {
					deconv = n
				}
			}
			if deconv.Int("group", 0) != c.channels {
				t.Errorf("group = %d, want %d", deconv.Int("group", 0), c.channels)
			}
			weight, err := back.Graph.Initializer(deconv.Inputs[1]).Float32s()
			if err != nil {
				t.Fatal(err)
			}
			ones := 0
			for _, v := range weight {
				if v == 1 {
					ones++
				} else if v != 0 {
					t.Fatalf("the weight holds %v, and is one-hot", v)
				}
			}
			if int64(ones) != c.channels*c.k() {
				t.Errorf("the weight has %d ones, want one per input channel, %d", ones, c.channels*c.k())
			}

			manifest := syntheticManifest("col2im-"+c.name, "x", 3, "y", 2+len(c.image))
			worst := 0.0
			for seed := range uint64(4) {
				input := c.input(seed + 1)
				want := runOnce(t, rt, manifest, original, input)
				got := runOnce(t, rt, manifest, rewritten, input)
				worst = max(worst, maxAbsDiff(want, got))
			}
			t.Logf("%s: max |Col2Im - ConvTranspose| over 4 random inputs = %g", c.name, worst)
			if worst > 1e-5 {
				t.Errorf("max abs difference %g, want <= 1e-5", worst)
			}
		})
	}
}

func TestAGraphTheRewriteCannotExpressIsRefusedAndLeftUnchanged(t *testing.T) {
	t.Parallel()

	base := col2imCases[0]
	dynamicImage := func() []byte {
		// image_shape is a graph input: computed at run time.
		g := message{}.
			sub(1, node("Col2Im", []string{"x", "image_shape", "block_shape"}, []string{"y"})).
			text(2, "col2im").
			sub(5, int64Initializer("block_shape", []int64{2, 2}, 2)).
			sub(11, valueInfo("x", protoFloat, 1, 4, 4)).
			sub(11, valueInfo("image_shape", protoInt64, 2)).
			sub(12, valueInfo("y", protoFloat, 1, 1, -1, -1))
		return modelAt(g, 8, 18)
	}
	notMultipleModel := func() []byte {
		g := message{}.
			sub(1, node("Col2Im", []string{"x", "image_shape", "block_shape"}, []string{"y"})).
			text(2, "col2im").
			sub(5, int64Initializer("image_shape", []int64{3, 3}, 2)).
			sub(5, int64Initializer("block_shape", []int64{2, 2}, 2)).
			sub(11, valueInfo("x", protoFloat, 1, 6, 4)).
			sub(12, valueInfo("y", protoFloat, 1, -1, 3, 3))
		return modelAt(g, 8, 18)
	}
	wrongBlocks := func() []byte {
		g := message{}.
			sub(1, node("Col2Im", []string{"x", "image_shape", "block_shape"}, []string{"y"})).
			text(2, "col2im").
			sub(5, int64Initializer("image_shape", []int64{3, 3}, 2)).
			sub(5, int64Initializer("block_shape", []int64{2, 2}, 2)).
			sub(11, valueInfo("x", protoFloat, 1, 4, 5)).
			sub(12, valueInfo("y", protoFloat, 1, 1, 3, 3))
		return modelAt(g, 8, 18)
	}
	unrecorded := col2imCases[2]

	for name, tc := range map[string]struct {
		body []byte
		opts rewrite.Col2ImOptions
		want string
	}{
		"image_shape computed at run time": {dynamicImage(), rewrite.Col2ImOptions{}, "computed at run time"},
		"channels not a multiple of K":     {notMultipleModel(), rewrite.Col2ImOptions{}, "not a multiple"},
		"blocks that do not match L":       {wrongBlocks(), rewrite.Col2ImOptions{}, "has 5 blocks"},
		"channels unknown":                 {unrecorded.model(), rewrite.Col2ImOptions{}, "InputChannels"},
		"a hint against the graph": {base.model(), rewrite.Col2ImOptions{InputChannels: map[string]int64{"x": 18}},
			"in the options"},
	} {
		m, err := graph.Load(tc.body)
		if err != nil {
			t.Fatalf("%s: Load: %v", name, err)
		}
		_, err = rewrite.Col2ImToConvTranspose(m, tc.opts)
		if !errors.Is(err, rewrite.ErrUnsupported) || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want ErrUnsupported saying %q", name, err, tc.want)
		}
		after, err := m.Marshal()
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(after, tc.body) {
			t.Errorf("%s: the refused model was changed", name)
		}
	}
}

func TestAModelWithoutCol2ImIsLeftAsItIs(t *testing.T) {
	t.Parallel()

	m, err := graph.Load(affineModel())
	if err != nil {
		t.Fatal(err)
	}
	report, err := rewrite.Col2ImToConvTranspose(m, rewrite.Col2ImOptions{})
	if err != nil || len(report.Rewritten) != 0 || report.NodesBefore != 3 || report.NodesAfter != 3 {
		t.Fatalf("report = %+v, err = %v", report, err)
	}
	out, _ := m.Marshal()
	if !bytes.Equal(out, affineModel()) {
		t.Error("the model was changed")
	}
}

// TestTheRewriteMatchesAHandComputedFold pins the arithmetic itself, so the
// comparison above cannot pass by both sides being wrong the same way.
//
// C = 1, block_shape [1, 2], image_shape [1, 3], default attributes: two
// blocks along the width. x [1, 2, 2] = {{a, b}, {c, d}}, channel j the block
// offset j and column l the block l. Block l covers columns l and l + 1, so
// y = {a, b + c, d}.
func TestTheRewriteMatchesAHandComputedFold(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	g := message{}.
		sub(1, node("Col2Im", []string{"x", "image_shape", "block_shape"}, []string{"y"})).
		text(2, "col2im").
		sub(5, int64Initializer("image_shape", []int64{1, 3}, 2)).
		sub(5, int64Initializer("block_shape", []int64{1, 2}, 2)).
		sub(11, valueInfo("x", protoFloat, 1, 2, 2)).
		sub(12, valueInfo("y", protoFloat, 1, 1, 1, 3))
	original := modelAt(g, 8, 18)
	m, err := graph.Load(original)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rewrite.Col2ImToConvTranspose(m, rewrite.Col2ImOptions{}); err != nil {
		t.Fatal(err)
	}
	rewritten, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	manifest := syntheticManifest("fold", "x", 3, "y", 4)
	input := onnx.Tensor{Shape: []int64{1, 2, 2}, Float32: []float32{1, 2, 30, 40}}
	for name, body := range map[string][]byte{"Col2Im": original, "ConvTranspose": rewritten} {
		y := runOnce(t, rt, manifest, body, input)
		assertShape(t, name, y.Shape, 1, 1, 1, 3)
		assertFloats(t, name, y.Float32, 1, 32, 40)
	}
}
