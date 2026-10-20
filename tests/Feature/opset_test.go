package feature_test

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/rewrite"
)

var opsetCases = []opsetCase{
	{
		// IR 3, as the exports that need the upgrade often are.
		name: "slice-squeeze-unsqueeze", opset: 7, ir: 3, input: []int64{2, 5, 6}, outRank: 4, converted: 3,
		nodes: []message{
			node("Slice", []string{"x"}, []string{"s"}, intsAttribute("starts", 1, 0), intsAttribute("ends", 2, 4), intsAttribute("axes", 0, 2)),
			node("Squeeze", []string{"s"}, []string{"q"}, intsAttribute("axes", 0)),
			node("Unsqueeze", []string{"q"}, []string{"y"}, intsAttribute("axes", 0, 3)),
		},
	},
	{
		name: "split-reducesum-topk", opset: 7, input: []int64{3, 5}, outRank: 2, converted: 3,
		nodes: []message{
			node("Split", []string{"x"}, []string{"a", "b"}, intAttribute("axis", 1), intsAttribute("split", 2, 3)),
			node("ReduceSum", []string{"a"}, []string{"r"}, intsAttribute("axes", 1), intAttribute("keepdims", 1)),
			node("TopK", []string{"b"}, []string{"v", "i"}, intAttribute("k", 2)),
			node("Concat", []string{"r", "v"}, []string{"y"}, intAttribute("axis", 1)),
		},
	},
	{
		name: "batchnorm-clip-pad-dropout", opset: 7, input: []int64{1, 2, 3, 3}, outRank: 4, converted: 5,
		nodes: []message{
			node("BatchNormalization", []string{"x", "scale", "bias", "mean", "var"}, []string{"n"},
				floatAttribute("epsilon", 1e-5), floatAttribute("momentum", 0.9), intAttribute("spatial", 1)),
			node("Clip", []string{"n"}, []string{"c"}, floatAttribute("min", -0.5), floatAttribute("max", 0.5)),
			node("Pad", []string{"c"}, []string{"p"}, intsAttribute("pads", 0, 0, 1, 0, 0, 0, 0, 2), floatAttribute("value", 0.25)),
			node("Pad", []string{"p"}, []string{"p2"}, stringAttribute("mode", "reflect"), intsAttribute("pads", 0, 0, 1, 1, 0, 0, 1, 1)),
			node("Dropout", []string{"p2"}, []string{"y"}, floatAttribute("ratio", 0.3)),
		},
		inits: []message{
			floatInitializer("scale", []float32{1.5, -0.5}, 2),
			floatInitializer("bias", []float32{0.1, 0.2}, 2),
			floatInitializer("mean", []float32{0.05, -0.1}, 2),
			floatInitializer("var", []float32{0.8, 1.3}, 2),
		},
	},
	{
		// Softmax along a middle axis becomes Flatten, Softmax, Reshape; along
		// the last it stays; after a Reshape whose rank the graph does not
		// tell, LogSoftmax takes the general form too.
		name: "softmax", opset: 7, input: []int64{2, 3, 4}, outRank: 3, converted: 4,
		nodes: []message{
			node("Softmax", []string{"x"}, []string{"a"}, intAttribute("axis", 1)),
			node("Softmax", []string{"a"}, []string{"b"}, intAttribute("axis", 2)),
			node("Shape", []string{"b"}, []string{"shape"}),
			node("Concat", []string{"shape"}, []string{"shape2"}, intAttribute("axis", 0)),
			node("Reshape", []string{"b", "shape2"}, []string{"r"}),
			node("LogSoftmax", []string{"r"}, []string{"c"}, intAttribute("axis", 1)),
			node("Hardmax", []string{"c"}, []string{"h"}, intAttribute("axis", -1)),
			node("Add", []string{"c", "h"}, []string{"y"}),
		},
	},
	{
		name: "upsample-7-nearest", opset: 7, input: []int64{1, 2, 3, 3}, outRank: 4, converted: 1,
		nodes: []message{node("Upsample", []string{"x"}, []string{"y"}, stringAttribute("mode", "nearest"), floatsAttribute("scales", 1, 1, 2, 3))},
	},
	{
		name: "upsample-9-linear", opset: 9, input: []int64{1, 1, 3, 4}, outRank: 4, converted: 1,
		nodes: []message{node("Upsample", []string{"x", "scales"}, []string{"y"}, stringAttribute("mode", "linear"))},
		inits: []message{floatInitializer("scales", []float32{1, 1, 2, 2}, 4)},
	},
	{
		name: "resize-10-nearest-up", opset: 10, input: []int64{1, 1, 3, 4}, outRank: 4, converted: 1,
		nodes: []message{node("Resize", []string{"x", "scales"}, []string{"y"}, stringAttribute("mode", "nearest"))},
		inits: []message{floatInitializer("scales", []float32{1, 1, 2, 3}, 4)},
	},
	{
		name: "resize-10-nearest-down", opset: 10, input: []int64{1, 1, 6, 8}, outRank: 4, converted: 1,
		nodes: []message{node("Resize", []string{"x", "scales"}, []string{"y"}, stringAttribute("mode", "nearest"))},
		inits: []message{floatInitializer("scales", []float32{1, 1, 0.5, 0.75}, 4)},
	},
	{
		name: "resize-10-linear", opset: 10, input: []int64{1, 1, 3, 4}, outRank: 4, converted: 1,
		nodes: []message{node("Resize", []string{"x", "scales"}, []string{"y"}, stringAttribute("mode", "linear"))},
		inits: []message{floatInitializer("scales", []float32{1, 1, 2, 2.5}, 4)},
	},
}

func TestUpgradeOpsetRewritesEveryChangedOperatorAndComputesTheSame(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})

	for _, c := range opsetCases {
		t.Run(c.name, func(t *testing.T) {
			body := c.model()
			out, report := upgraded(t, body)
			if report.OpsetBefore != c.opset || report.OpsetAfter != 13 || report.IRAfter != 7 || len(report.Converted) != c.converted {
				t.Fatalf("report = %+v", report)
			}
			m, err := graph.Load(out)
			if err != nil {
				t.Fatal(err)
			}
			if v, _ := m.Opset(graph.DefaultDomain); v != 13 || m.IRVersion != 7 {
				t.Fatalf("written at opset %d, IR %d", v, m.IRVersion)
			}
			for _, n := range m.Graph.Nodes {
				for _, a := range n.Attributes {
					switch a.Name {
					case "axes", "starts", "ends", "split", "k", "min", "max", "pads", "value", "ratio", "spatial", "scales":
						t.Errorf("%s %q still has the attribute %q", n.OpType, n.Name, a.Name)
					}
				}
				if n.OpType == "Upsample" {
					t.Errorf("Upsample %q is left", n.Name)
				}
			}
			diff := sameOnCPU(t, rt, c.manifest("old"), c.manifest("new"), body, out, c.input)
			if diff > 1e-6 {
				t.Errorf("the upgraded model differs from the original by %g", diff)
			}
		})
	}
}

func TestUpgradeOpsetKeepsTheTwoDimensionalSoftmaxOfOldOpsets(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})

	// Softmax-1 at axis 1 of a [1, 2, 2] tensor of zeros coerces it to
	// [1, 4] and gives 1/4 everywhere; Softmax-13 at axis 1 would give 1/2.
	c := opsetCase{
		name: "softmax-pinned", opset: 7, input: []int64{1, 2, 2}, outRank: 3,
		nodes: []message{node("Softmax", []string{"x"}, []string{"y"}, intAttribute("axis", 1))},
	}
	out, report := upgraded(t, c.model())
	if len(report.Converted) != 1 || len(report.Converted[0].Nodes) != 4 {
		t.Fatalf("converted = %+v", report.Converted)
	}
	got := runOnce(t, rt, c.manifest("pinned"), out, onnx.Tensor{Shape: []int64{1, 2, 2}, Float32: make([]float32, 4)})
	for _, v := range got.Float32 {
		if math.Abs(float64(v)-0.25) > 1e-7 {
			t.Fatalf("softmax of zeros = %v, want 0.25 everywhere", got.Float32)
		}
	}
}

func TestUpgradeOpsetRefusesWhatItDoesNotHandleAndLeavesTheModel(t *testing.T) {
	for name, tc := range map[string]struct {
		c    opsetCase
		opts rewrite.OpsetOptions
		want string
	}{
		"below opset 7": {c: opsetCase{name: "old", opset: 6, input: []int64{2}, outRank: 1,
			nodes: []message{node("Relu", []string{"x"}, []string{"y"})}}, want: "opset 7 or later"},
		"an operator not enumerated": {c: opsetCase{name: "lstm", opset: 7, input: []int64{2}, outRank: 1,
			nodes: []message{node("Relu", []string{"x"}, []string{"r"}), node("Scatter", []string{"r", "r", "r"}, []string{"y"})}}, want: "does not handle Scatter"},
		"non-spatial batch normalization": {c: opsetCase{name: "bn", opset: 7, input: []int64{1, 2}, outRank: 2,
			nodes: []message{node("BatchNormalization", []string{"x", "s", "s", "s", "s"}, []string{"y"}, intAttribute("spatial", 0))},
			inits: []message{floatInitializer("s", []float32{1, 1}, 2)}}, want: "spatial = 0"},
		"nearest resize up and down at once": {c: opsetCase{name: "resize", opset: 10, input: []int64{1, 1, 4, 4}, outRank: 4,
			nodes: []message{node("Resize", []string{"x", "scales"}, []string{"y"})},
			inits: []message{floatInitializer("scales", []float32{1, 1, 2, 0.5}, 4)}}, want: "enlarge one axis and shrink another"},
		"nearest resize with computed scales": {c: opsetCase{name: "resize", opset: 10, input: []int64{4}, outRank: 1,
			nodes: []message{node("Abs", []string{"x"}, []string{"scales"}), node("Resize", []string{"x", "scales"}, []string{"y"})}}, want: "computed at run time"},
		"same padding of a transposed convolution": {c: opsetCase{name: "deconv", opset: 10, input: []int64{1, 1, 4, 4}, outRank: 4,
			nodes: []message{node("ConvTranspose", []string{"x", "w"}, []string{"y"}, stringAttribute("auto_pad", "SAME_UPPER"), intsAttribute("strides", 2, 2))},
			inits: []message{floatInitializer("w", make([]float32, 9), 1, 1, 3, 3)}}, want: "redefined at opset 11"},
		"same padding of a strided convolution": {c: opsetCase{name: "conv", opset: 9, input: []int64{1, 1, 4, 4}, outRank: 4,
			nodes: []message{node("Conv", []string{"x", "w"}, []string{"y"}, stringAttribute("auto_pad", "SAME_LOWER"), intsAttribute("strides", 2, 2))},
			inits: []message{floatInitializer("w", make([]float32, 9), 1, 1, 3, 3)}}, want: "redefined at opset 11"},
		"an unsqueeze without axes": {c: opsetCase{name: "unsqueeze", opset: 7, input: []int64{2}, outRank: 2,
			nodes: []message{node("Unsqueeze", []string{"x"}, []string{"y"})}}, want: "axes attribute is required"},
		"a clip on a tensor of unknown type": {c: opsetCase{name: "clip", opset: 7, input: []int64{2}, outRank: 1,
			nodes: []message{node("Cast", []string{"x"}, []string{"z"}, intAttribute("to", 6)),
				node("Clip", []string{"z"}, []string{"y"}, floatAttribute("min", 0))}}, want: "only float32 is written"},
		"a target other than 13": {c: opsetCase{name: "target", opset: 7, input: []int64{2}, outRank: 1,
			nodes: []message{node("Relu", []string{"x"}, []string{"y"})}}, opts: rewrite.OpsetOptions{Target: 17}, want: "not a target"},
	} {
		t.Run(name, func(t *testing.T) {
			body := tc.c.model()
			m, err := graph.Load(body)
			if err != nil {
				t.Fatal(err)
			}
			_, err = rewrite.UpgradeOpset(m, tc.opts)
			if !errors.Is(err, rewrite.ErrUnsupported) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("UpgradeOpset = %v, want ErrUnsupported naming %q", err, tc.want)
			}
			out, err := m.Marshal()
			if err != nil || !bytes.Equal(out, body) {
				t.Errorf("a refused model was changed (%v)", err)
			}
		})
	}
}

func TestUpgradeOpsetLeavesAModelAlreadyAtTheTarget(t *testing.T) {
	body := convNet(13)
	out, report := upgraded(t, body)
	if report.OpsetBefore != 13 || report.OpsetAfter != 13 || len(report.Converted) != 0 || !bytes.Equal(out, body) {
		t.Errorf("report = %+v, unchanged = %v", report, bytes.Equal(out, body))
	}
}
