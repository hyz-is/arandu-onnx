package unit_test

import (
	"os"
	"path/filepath"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
)

// The fixtures are described in testdata/placement/README.md: the darwin ones
// are captured, the linux ones are written in the format of ONNX Runtime
// 1.30.0's source because no NVIDIA machine was available.

func placementFixture(t *testing.T, name string) onnx.Placement {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(packageRoot(t), "tests", "Unit", "testdata", "placement", name))
	if err != nil {
		t.Fatalf("reading %s: %v", name, err)
	}
	return onnx.ParsePlacement(string(body))
}

func TestACapturedCPUOnlyPlacementIsRead(t *testing.T) {
	t.Parallel()

	p := placementFixture(t, "darwin-cpu-all.log")
	if !p.Captured || p.Nodes[onnx.LogProviderCPU] != 3 || len(p.Nodes) != 1 {
		t.Fatalf("placement = %+v, want 3 nodes on the CPU", p)
	}
	// A graph wholly on the CPU is a fallback for an accelerated session.
	if got := p.Fallback(onnx.DefaultCPUOps()); got["every node"] != 3 {
		t.Errorf("Fallback = %v, want every node", got)
	}
}

func TestACapturedMixedPlacementIsRead(t *testing.T) {
	t.Parallel()

	// Captured from CoreML: the fused node's operator starts with a digit, and
	// the section is followed by a warning and by lines that name providers.
	p := placementFixture(t, "darwin-coreml-mixed.log")
	if !p.Captured || p.Nodes[onnx.LogProviderCoreML] != 1 || p.Nodes[onnx.LogProviderCPU] != 1 || len(p.Nodes) != 2 {
		t.Fatalf("placement = %+v", p)
	}
	if p.CPUOps["Cast"] != 1 || len(p.CPUOps) != 1 {
		t.Errorf("CPUOps = %v, want Cast x1", p.CPUOps)
	}
	if got := p.Fallback(onnx.DefaultCPUOps()); got != nil {
		t.Errorf("Fallback = %v, want none: Cast is shape arithmetic", got)
	}
	if got := p.Fallback(nil); got["Cast"] != 1 {
		t.Errorf("Fallback with no allowed operator = %v, want Cast", got)
	}
}

func TestATensorRTPlacementCountsEnginesAndKeepsReadingAfterThem(t *testing.T) {
	t.Parallel()

	// The engine names carry the graph's name, here with '-' in it, and the
	// CPU block comes after them: an engine line the parser did not
	// recognise would end the section before the CPU nodes were read.
	p := placementFixture(t, "linux-tensorrt-mixed.log")
	want := map[string]int{onnx.LogProviderTensorRT: 2, onnx.LogProviderCUDA: 3, onnx.LogProviderCPU: 4}
	for provider, n := range want {
		if p.Nodes[provider] != n {
			t.Errorf("Nodes[%s] = %d, want %d (%+v)", provider, p.Nodes[provider], n, p)
		}
	}
	if len(p.CPUOps) != 4 {
		t.Errorf("CPUOps = %v, want the four shape operators", p.CPUOps)
	}
	if got := p.Fallback(onnx.DefaultCPUOps()); got != nil {
		t.Errorf("Fallback = %v, want none", got)
	}
	if got := p.String(); got != "CPUExecutionProvider=4 CUDAExecutionProvider=3 TensorrtExecutionProvider=2" {
		t.Errorf("String = %q", got)
	}

	all := placementFixture(t, "linux-tensorrt-all.log")
	if all.Nodes[onnx.LogProviderTensorRT] != 1 || all.Fallback(onnx.DefaultCPUOps()) != nil {
		t.Errorf("a graph in one engine = %+v", all)
	}
}

func TestAModelQuantizedForTheCPUIsAFallback(t *testing.T) {
	t.Parallel()

	p := placementFixture(t, "linux-cuda-qoperator-fallback.log")
	got := p.Fallback(onnx.DefaultCPUOps())
	if len(got) != 2 || got["QLinearConv"] != 2 || got["NhwcMaxPool"] != 1 {
		t.Errorf("Fallback = %v, want QLinearConv x2 and NhwcMaxPool x1", got)
	}
	if p.Nodes[onnx.LogProviderCUDA] != 2 {
		t.Errorf("the CUDA block after the CPU one was not read: %+v", p)
	}
}

func TestALogWithoutPlacementIsNotACleanOne(t *testing.T) {
	t.Parallel()

	p := onnx.ParsePlacement("2026-10-19 [I:onnxruntime:, inference_session.cc:2049 Initialize] Initializing session.\n")
	if p.Captured || p.Fallback(nil) != nil || p.String() != "not captured" {
		t.Errorf("a log with no placement = %+v", p)
	}
}
