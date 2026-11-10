package feature_test

import (
	"errors"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
)

func TestABoundSessionRunsOnTheMemoryItHandsOut(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())

	bound, err := runtime.OpenBound(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model bound: %v", err)
	}

	x := bound.InputFloat32("x")
	if len(x) != 3 {
		t.Fatalf("the input holds %d values, want 3: the dynamic first dimension is bound as 1", len(x))
	}
	y := bound.OutputFloat32("y")
	label := bound.OutputInt64("label")
	if len(y) != 2 || len(label) != 1 {
		t.Fatalf("outputs hold %d and %d values, want 2 and 1", len(y), len(label))
	}

	copy(x, []float32{1, 0, 0})
	if err := bound.Run(); err != nil {
		t.Fatalf("running: %v", err)
	}
	assertFloats(t, "y", y, 1.5, 1.5)

	// The second run overwrites the same memory: nothing was allocated for it.
	copy(x, []float32{0, 1, 1})
	if err := bound.Run(); err != nil {
		t.Fatalf("running again: %v", err)
	}
	assertFloats(t, "y", y, 8.5, 9.5)
	if label[0] != 1 {
		t.Errorf("label = %d, want 1", label[0])
	}
	if bound.InputFloat32("missing") != nil || bound.InputUint8("x") != nil {
		t.Error("an accessor returned memory for a tensor that is not of that name and type")
	}
	if bound.Model().Provider != onnx.ProviderCPU {
		t.Errorf("provider = %q, want cpu", bound.Model().Provider)
	}
}

func TestABoundSessionAllocationsPerRunAreNone(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	bound, err := runtime.OpenBound(manifest, dir)
	if err != nil {
		t.Fatalf("opening: %v", err)
	}
	defer bound.Close()
	copy(bound.InputFloat32("x"), []float32{1, 2, 3})

	// The Go side of Run builds two small slices of pointers for the C call.
	// The tensors and the result are never allocated again.
	allocs := testing.AllocsPerRun(50, func() {
		if err := bound.Run(); err != nil {
			t.Fatalf("running: %v", err)
		}
	})
	if allocs > 8 {
		t.Errorf("a run allocates %.0f times, want a few small slices and no tensors", allocs)
	}
}

func TestABoundSessionIsClosedOnce(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	bound, err := runtime.OpenBound(manifest, dir)
	if err != nil {
		t.Fatalf("opening: %v", err)
	}
	if err := bound.Close(); err != nil {
		t.Fatalf("closing: %v", err)
	}
	if err := bound.Close(); err != nil {
		t.Fatalf("closing twice: %v", err)
	}
	if err := bound.Run(); !errors.Is(err, onnx.ErrClosed) {
		t.Errorf("Run after Close = %v, want ErrClosed", err)
	}
}

func TestABoundSessionNeedsFixedShapes(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest := affineManifest()
	manifest.Inputs = []onnx.TensorSpec{{Name: "x", ElementType: onnx.Float32, Shape: []int64{-1, -1}}}
	manifest, dir := writeModel(t, manifest, affineModel())
	if _, err := runtime.OpenBound(manifest, dir); !errors.Is(err, onnx.ErrInvalidManifest) {
		t.Errorf("a dynamic inner dimension = %v, want ErrInvalidManifest", err)
	}
}

func TestAClosedRuntimeLeavesABoundSessionSafeToClose(t *testing.T) {
	runtime, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Init(); err != nil {
		t.Skipf("ONNX Runtime is not available here: %v", err)
	}
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	bound, err := runtime.OpenBound(manifest, dir)
	if err != nil {
		t.Fatalf("opening: %v", err)
	}
	if err := runtime.Close(); err != nil {
		t.Fatalf("closing the runtime: %v", err)
	}
	if err := bound.Close(); err != nil {
		t.Errorf("closing after the runtime: %v", err)
	}
}

func TestQNNWithoutItsPluginIsReportedAndRefusedWhenRequired(t *testing.T) {
	cfg := onnx.Config{
		Providers: []string{onnx.ProviderQNN, onnx.ProviderCPU},
		QNN:       onnx.QNNOptions{LibraryPath: "/nonexistent/libqnn_plugin.so", BackendPath: "libQnnHtp.so"},
	}
	runtime := newRuntime(t, cfg)
	var qnn *onnx.ProviderStatus
	for _, s := range runtime.Providers() {
		if s.Name == onnx.ProviderQNN {
			s := s
			qnn = &s
		}
	}
	if qnn == nil || qnn.Available || qnn.Reason == "" {
		t.Fatalf("qnn status = %+v, want unavailable with a reason", qnn)
	}

	// Without the guard the session falls to the CPU, as every preference does.
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening on the CPU: %v", err)
	}
	if session.Model().Provider != onnx.ProviderCPU {
		t.Errorf("provider = %q, want cpu", session.Model().Provider)
	}

	// With it, the session is refused.
	cfg.Placement.RequireAccelerated = true
	guarded := newRuntime(t, cfg)
	if _, err := guarded.OpenBound(manifest, dir); !errors.Is(err, onnx.ErrCPUFallback) {
		t.Errorf("OpenBound with RequireAccelerated = %v, want ErrCPUFallback", err)
	}
}
