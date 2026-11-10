package feature_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/rewrite"
)

// poolModel is x [1, 3, 6, 6] -> AveragePool -> y, with the input's shape
// recorded unless unrecorded is set.
func poolModel(unrecorded bool, attributes ...message) []byte {
	in := valueInfo("x", protoFloat, 1, 3, 6, 6)
	if unrecorded {
		in = valueInfo("x", protoFloat, 1, -1, 6, 6)
	}
	g := message{}.
		sub(1, node("AveragePool", []string{"x"}, []string{"y"}, attributes...)).
		text(2, "pool").
		sub(11, in).
		sub(12, valueInfo("y", protoFloat, -1, -1, -1, -1))
	return modelAt(g, 8, 17)
}

func TestAveragePoolIsRewrittenIntoADepthwiseConvThatComputesTheSame(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	manifest := syntheticManifest("pool", "x", 4, "y", 4)

	for name, attributes := range map[string][]message{
		"2x2 stride 1":        {intsAttribute("kernel_shape", 2, 2), intsAttribute("strides", 1, 1)},
		"3x3 stride 2":        {intsAttribute("kernel_shape", 3, 3), intsAttribute("strides", 2, 2)},
		"2x3 default strides": {intsAttribute("kernel_shape", 2, 3)},
		"explicit zero pads and VALID": {
			intsAttribute("kernel_shape", 2, 2), intsAttribute("pads", 0, 0, 0, 0), stringAttribute("auto_pad", "VALID"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			body := poolModel(false, attributes...)
			m, err := graph.Load(body)
			if err != nil {
				t.Fatal(err)
			}
			report, err := rewrite.AveragePoolToDepthwiseConv(m, rewrite.AveragePoolOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Rewritten) != 1 || report.Rewritten[0].Channels != 3 || report.NodesBefore != 1 || report.NodesAfter != 1 {
				t.Fatalf("report = %+v", report)
			}
			counts := m.Graph.OpCounts()
			if counts["AveragePool"] != 0 || counts["Conv"] != 1 || len(m.Graph.Initializers) != 1 {
				t.Fatalf("ops %v, %d initializers", counts, len(m.Graph.Initializers))
			}
			weight := m.Graph.Initializers[0]
			values, err := weight.Float32s()
			if err != nil {
				t.Fatal(err)
			}
			kernel := report.Rewritten[0].Kernel
			if len(weight.Dims) != 4 || weight.Dims[0] != 3 || weight.Dims[1] != 1 || weight.Dims[2] != kernel[0] || weight.Dims[3] != kernel[1] {
				t.Fatalf("weight dims %v for kernel %v", weight.Dims, kernel)
			}
			for _, v := range values {
				if want := 1 / float32(kernel[0]*kernel[1]); v != want {
					t.Fatalf("weight value %v, want %v", v, want)
				}
			}
			out := marshal(t, m)
			if diff := sameOnCPU(t, rt, manifest, manifest, body, out, []int64{1, 3, 6, 6}); diff > 1e-6 {
				t.Errorf("the rewritten model differs from the original by %g", diff)
			}
		})
	}
}

func TestAveragePoolChannelsComeFromAHintWhenTheGraphDoesNotRecordThem(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	manifest := syntheticManifest("pool", "x", 4, "y", 4)
	body := poolModel(true, intsAttribute("kernel_shape", 2, 2))

	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rewrite.AveragePoolToDepthwiseConv(m, rewrite.AveragePoolOptions{}); !errors.Is(err, rewrite.ErrUnsupported) || !strings.Contains(err.Error(), "channels") {
		t.Fatalf("without a hint: %v", err)
	}
	if _, err := rewrite.AveragePoolToDepthwiseConv(m, rewrite.AveragePoolOptions{InputChannels: map[string]int64{"x": 3}}); err != nil {
		t.Fatal(err)
	}
	if diff := sameOnCPU(t, rt, manifest, manifest, body, marshal(t, m), []int64{1, 3, 6, 6}); diff > 1e-6 {
		t.Errorf("the rewritten model differs from the original by %g", diff)
	}
}

func TestAnAveragePoolTheIdentityDoesNotCoverIsRefusedAndLeftUnchanged(t *testing.T) {
	for name, tc := range map[string]struct {
		attributes []message
		hint       map[string]int64
		want       string
	}{
		"padded":           {[]message{intsAttribute("kernel_shape", 3, 3), intsAttribute("pads", 1, 1, 1, 1)}, nil, "pads"},
		"ceil mode":        {[]message{intsAttribute("kernel_shape", 2, 2), intAttribute("ceil_mode", 1)}, nil, "ceil_mode"},
		"dilated":          {[]message{intsAttribute("kernel_shape", 2, 2), intsAttribute("dilations", 2, 2)}, nil, "dilations"},
		"SAME auto_pad":    {[]message{intsAttribute("kernel_shape", 2, 2), stringAttribute("auto_pad", "SAME_UPPER")}, nil, "auto_pad"},
		"no kernel":        {nil, nil, "kernel_shape"},
		"wrong rank":       {[]message{intsAttribute("kernel_shape", 2)}, nil, "rank"},
		"hint disagrees":   {[]message{intsAttribute("kernel_shape", 2, 2)}, map[string]int64{"x": 4}, "hint"},
		"strides mismatch": {[]message{intsAttribute("kernel_shape", 2, 2), intsAttribute("strides", 1)}, nil, "strides"},
	} {
		t.Run(name, func(t *testing.T) {
			body := poolModel(false, tc.attributes...)
			m, err := graph.Load(body)
			if err != nil {
				t.Fatal(err)
			}
			_, err = rewrite.AveragePoolToDepthwiseConv(m, rewrite.AveragePoolOptions{InputChannels: tc.hint})
			if !errors.Is(err, rewrite.ErrUnsupported) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("AveragePoolToDepthwiseConv = %v, want ErrUnsupported naming %q", err, tc.want)
			}
			if !bytes.Equal(marshal(t, m), body) {
				t.Error("a refused model was changed")
			}
		})
	}
}

func TestAModelWithoutAveragePoolIsLeftAsItIsByThePoolRewrite(t *testing.T) {
	body := affineModel()
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	report, err := rewrite.AveragePoolToDepthwiseConv(m, rewrite.AveragePoolOptions{})
	if err != nil || len(report.Rewritten) != 0 || report.NodesBefore != report.NodesAfter {
		t.Fatalf("report %+v, err %v", report, err)
	}
	if !bytes.Equal(marshal(t, m), body) {
		t.Error("the model changed")
	}
}

func TestTheRewriteCommandRewritesAveragePoolOnlyWhenAsked(t *testing.T) {
	rt := newRuntime(t, onnx.Config{})
	binary := buildCommand(t, "arandu-onnx-rewrite")

	source := syntheticManifest("pool", "x", 4, "y", 4)
	source.Inputs[0].Shape = []int64{1, 3, 6, 6}
	manifest, dir := writeModel(t, source, poolModel(false, intsAttribute("kernel_shape", 2, 2), intsAttribute("strides", 1, 1)))
	manifestPath := writeManifest(t, manifest, dir)
	out := filepath.Join(dir, "pool.depthwise.onnx")

	// Off by default: nothing else is asked for, so the command refuses.
	if _, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-library", rt.Config().SharedLibraryPath,
		"-out", out); err == nil || !strings.Contains(stderr, "none of the rewrites") {
		t.Fatalf("without the flag: %v, %s", err, stderr)
	}

	stdout, stderr, err := runCommand(t, binary, "-manifest", manifestPath, "-library", rt.Config().SharedLibraryPath,
		"-average-pool-to-depthwise-conv", "-tolerance", "1e-6", "-out", out)
	if err != nil {
		t.Fatalf("running the command: %v\n%s", err, stderr)
	}
	printed, err := onnx.LoadManifest(stdout)
	if err != nil {
		t.Fatalf("the printed manifest %s: %v", stdout, err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	m, err := graph.Load(body)
	if err != nil {
		t.Fatal(err)
	}
	if printed.SHA256 != sha(body) || printed.DerivedFrom == nil || printed.DerivedFrom.ID != "pool" ||
		!strings.Contains(printed.DerivedFrom.Transform, "AveragePool") {
		t.Errorf("printed = %+v, derived from %+v", printed, printed.DerivedFrom)
	}
	if counts := m.Graph.OpCounts(); counts["AveragePool"] != 0 || counts["Conv"] != 1 {
		t.Errorf("ops %v", counts)
	}
}
