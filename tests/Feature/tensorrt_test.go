package feature_test

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
)

// nvidiaProviders is the order an NVIDIA machine is configured with.
var nvidiaProviders = []string{onnx.ProviderTensorRT, onnx.ProviderCUDA, onnx.ProviderCPU}

// available reports whether the runtime found a provider.
func available(rt *onnx.Runtime, name string) (bool, string) {
	for _, status := range rt.Providers() {
		if status.Name == name {
			return status.Available, status.Reason
		}
	}
	return false, "not configured"
}

func TestAMissingTensorRTFallsBackAndSaysWhy(t *testing.T) {
	rt := newRuntime(t, onnx.Config{Providers: nvidiaProviders, TensorRT: onnx.TensorRTOptions{
		CacheDir: t.TempDir(), INT8: true, FP16: true, EngineCache: true,
	}})
	if ok, _ := available(rt, onnx.ProviderTensorRT); ok {
		t.Skip("TensorRT is available here; TestTensorRTRunsTheCalibratedModelInINT8 covers this machine")
	}
	if _, reason := available(rt, onnx.ProviderTensorRT); reason == "" {
		t.Error("TensorRT is unavailable and Providers gives no reason")
	}

	manifest, dir, _ := withTable(t, rt, calibration.FormatONNXRuntime)
	session, err := rt.Open(manifest, dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if info := session.Model(); info.Provider != onnx.ProviderCPU || info.Calibration != "" || info.Device != 0 {
		t.Errorf("ModelInfo = %+v, want the CPU, no calibration and no device", info)
	}
	if _, err := session.Run(context.Background(), affineSamples()[0]); err != nil {
		t.Errorf("Run: %v", err)
	}
}

func TestRequireAcceleratedRefusesTheCPU(t *testing.T) {
	rt := newRuntime(t, onnx.Config{
		Providers: nvidiaProviders,
		Placement: onnx.PlacementOptions{RequireAccelerated: true},
	})
	if ok, _ := available(rt, onnx.ProviderTensorRT); ok {
		t.Skip("TensorRT is available here, so nothing falls back to the CPU")
	}
	if ok, _ := available(rt, onnx.ProviderCUDA); ok {
		t.Skip("CUDA is available here, so nothing falls back to the CPU")
	}
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	if _, err := rt.Open(manifest, dir); !errors.Is(err, onnx.ErrCPUFallback) {
		t.Fatalf("Open = %v, want ErrCPUFallback", err)
	}
	if _, err := rt.OpenOn(manifest, dir, onnx.MaxDevice+1); err == nil {
		t.Error("OpenOn accepted a device past MaxDevice")
	}
}

func TestThePlacementOfTheFirstSessionIsCaptured(t *testing.T) {
	rt := newRuntime(t, onnx.Config{Placement: onnx.PlacementOptions{Capture: true}})
	manifest, dir := writeModel(t, affineManifest(), affineModel())

	first, err := rt.Open(manifest, dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	p := first.Placement()
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" && !p.Captured {
		t.Skipf("standard error is not captured on %s", runtime.GOOS)
	}
	if !p.Captured || p.Nodes[onnx.LogProviderCPU] == 0 || len(p.Nodes) != 1 {
		t.Fatalf("Placement = %+v, want every node on the CPU, captured from ONNX Runtime's log", p)
	}
	t.Logf("captured: %s", p)

	second, err := rt.Open(manifest, dir)
	if err != nil {
		t.Fatalf("second Open: %v", err)
	}
	if got := second.Placement(); got.String() != p.String() {
		t.Errorf("second session's placement = %s, want the first's %s", got, p)
	}
	if _, err := second.Run(context.Background(), affineSamples()[0]); err != nil {
		t.Errorf("Run after a captured build: %v", err)
	}
}

func TestCoreMLPlacementIsCaptured(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Core ML exists on Apple platforms only")
	}
	rt := newRuntime(t, onnx.Config{
		Providers: []string{onnx.ProviderCoreML, onnx.ProviderCPU},
		Placement: onnx.PlacementOptions{Capture: true},
	})
	if ok, reason := available(rt, onnx.ProviderCoreML); !ok {
		t.Skipf("Core ML is not in this ONNX Runtime: %s", reason)
	}
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := rt.Open(manifest, dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	p := session.Placement()
	if !p.Captured {
		t.Fatalf("Placement = %+v, want it captured", p)
	}
	t.Logf("Core ML session: %s, CPU operators %v", p, p.CPUOps)
}

// TestTensorRTRunsTheCalibratedModelInINT8 is the test that needs an NVIDIA
// card, the GPU build of ONNX Runtime and the TensorRT libraries. Everywhere
// else it skips, naming what is missing; on a GPU node it is the evidence
// that the table reaches TensorRT.
func TestTensorRTRunsTheCalibratedModelInINT8(t *testing.T) {
	cache := t.TempDir()
	rt := newRuntime(t, onnx.Config{
		Providers: nvidiaProviders,
		TensorRT: onnx.TensorRTOptions{
			CacheDir: cache, INT8: true, FP16: true, EngineCache: true, TimingCache: true,
		},
		Placement: onnx.PlacementOptions{Capture: true},
	})
	if ok, reason := available(rt, onnx.ProviderTensorRT); !ok {
		t.Skipf("TensorRT is not available on this machine, so INT8 on TensorRT cannot be tested here: %s", reason)
	}

	manifest, dir, table := withTable(t, rt, calibration.FormatONNXRuntime)
	session, err := rt.Open(manifest, dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	info := session.Model()
	if info.Provider != onnx.ProviderTensorRT || info.Calibration != manifest.Calibration.SHA256 {
		t.Fatalf("ModelInfo = %+v, want tensorrt with the table's digest", info)
	}
	plan, err := rt.Config().Plan(manifest, rt.Config().Device)
	if err != nil {
		t.Fatal(err)
	}
	written, err := os.ReadFile(plan.CalibrationPath)
	if err != nil || !bytes.Equal(written, table) {
		t.Fatalf("the table TensorRT reads at %s is not the verified one: %v", plan.CalibrationPath, err)
	}
	if info, err := os.Stat(filepath.Dir(plan.CalibrationPath)); err != nil || info.Mode().Perm() != 0o700 {
		t.Errorf("the engine cache directory is %v, %v; want private to the process", info, err)
	}
	p := session.Placement()
	if !p.Captured || p.Nodes[onnx.LogProviderTensorRT] == 0 {
		t.Errorf("Placement = %s, want nodes on TensorRT", p)
	}
	result, err := session.Run(context.Background(), affineSamples()[0])
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	t.Logf("TensorRT INT8 y = %v (FP32 reference 1.5 1.5 8.5 9.5), placement %s", result.Outputs["y"].Float32, p)
}

func TestConcurrentOpensWhileCapturingAgreeOnThePlacement(t *testing.T) {
	rt := newRuntime(t, onnx.Config{Placement: onnx.PlacementOptions{Capture: true}})
	affine, dir := writeModel(t, affineManifest(), affineModel())
	cast, castDir := writeModel(t, castManifest(), castModel())

	var wg sync.WaitGroup
	placements := make([]string, 16)
	errs := make([]error, 16)
	for i := range placements {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m, d := affine, dir
			if i%2 == 1 {
				m, d = cast, castDir
			}
			session, err := rt.Open(m, d)
			if err != nil {
				errs[i] = err
				return
			}
			placements[i] = session.Placement().String()
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("Open %d: %v", i, err)
		}
	}
	for i := 2; i < len(placements); i++ {
		if placements[i] != placements[i%2] {
			t.Errorf("session %d placement %q, the first of its model had %q", i, placements[i], placements[i%2])
		}
	}
	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		if placements[0] == "not captured" || placements[1] == "not captured" {
			t.Errorf("placements = %v, want both models captured", placements[:2])
		}
	}
}
