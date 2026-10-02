package feature_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// These tests load real models into the ONNX Runtime library on this machine.
// They skip, saying why, where the library is absent.

func TestAVerifiedModelRunsAndReturnsItsOutputs(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())

	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}

	result, err := session.Run(context.Background(), map[string]onnx.Tensor{
		"x": {Shape: []int64{2, 3}, Float32: []float32{1, 0, 0, 0, 1, 1}},
	})
	if err != nil {
		t.Fatalf("running the model: %v", err)
	}

	// Row one picks the first row of w, row two the sum of the last two.
	y := result.Outputs["y"]
	assertShape(t, "y", y.Shape, 2, 2)
	assertFloats(t, "y", y.Float32, 1.5, 1.5, 8.5, 9.5)

	label := result.Outputs["label"]
	assertShape(t, "label", label.Shape, 2)
	if len(label.Int64) != 2 || label.Int64[0] != 0 || label.Int64[1] != 1 {
		t.Errorf("label = %v, want [0 1]: argmax returns the first of two equal values", label.Int64)
	}
}

func TestEveryResultCarriesTheModelItCameFrom(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())

	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}
	result, err := session.Run(context.Background(), map[string]onnx.Tensor{
		"x": {Shape: []int64{1, 3}, Float32: []float32{0, 0, 0}},
	})
	if err != nil {
		t.Fatalf("running the model: %v", err)
	}

	want := onnx.ModelInfo{
		ID:        "test.affine",
		Version:   "1.0.0",
		SHA256:    manifest.SHA256,
		Runtime:   "onnxruntime " + runtime.Version(),
		Precision: onnx.PrecisionFP32,
		Provider:  onnx.ProviderCPU,
		Status:    onnx.StatusApproved,
	}
	if result.Model != want {
		t.Errorf("result.Model = %+v, want %+v", result.Model, want)
	}
	if session.Model() != want {
		t.Errorf("session.Model() = %+v, want %+v", session.Model(), want)
	}
	if !strings.HasPrefix(runtime.Version(), "1.") {
		t.Errorf("Version() = %q, want the library's own version string", runtime.Version())
	}
}

func TestUint8InputsAndCastsRun(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, castManifest(), castModel())

	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}
	result, err := session.Run(context.Background(), map[string]onnx.Tensor{
		"u": {Shape: []int64{4}, Uint8: []uint8{0, 1, 128, 255}},
	})
	if err != nil {
		t.Fatalf("running the model: %v", err)
	}
	assertFloats(t, "f", result.Outputs["f"].Float32, 0, 1, 128, 255)
}

func TestAFileWhoseDigestIsNotTheManifestsIsNeverLoaded(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())

	// One byte of the file changes after the digest was recorded: the
	// producer name, so the graph itself would still load.
	path := filepath.Join(dir, manifest.File)
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	at := strings.Index(string(body), "arandu-onnx-tests")
	body[at] = 'A'
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}

	session, err := runtime.Open(manifest, dir)
	if !errors.Is(err, onnx.ErrDigestMismatch) {
		t.Fatalf("Open = %v, %v; want ErrDigestMismatch", session, err)
	}
	if !strings.Contains(err.Error(), manifest.SHA256) {
		t.Errorf("the refusal does not name the recorded digest: %v", err)
	}
}

func TestABlockedModelIsRefusedBeforeItsFileIsRead(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, _ := writeModel(t, affineManifest(), affineModel())
	manifest.Status = onnx.StatusBlocked

	// The directory does not exist, so a refusal for any other reason than
	// the status would be a read error.
	_, err := runtime.Open(manifest, filepath.Join(t.TempDir(), "absent"))
	if !errors.Is(err, onnx.ErrModelBlocked) {
		t.Fatalf("Open = %v, want ErrModelBlocked", err)
	}
}

func TestADevOnlyModelLoadsAndSaysSo(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	manifest.Status = onnx.StatusDevOnly

	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}
	if session.Model().Status != onnx.StatusDevOnly {
		t.Errorf("Model().Status = %q, want dev-only", session.Model().Status)
	}
}

func TestAManifestThatDoesNotDescribeTheGraphIsRefused(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})

	for name, change := range map[string]func(*onnx.Manifest){
		"an input name":         func(m *onnx.Manifest) { m.Inputs[0].Name = "image" },
		"an input element type": func(m *onnx.Manifest) { m.Inputs[0].ElementType = onnx.Uint8 },
		"an input rank":         func(m *onnx.Manifest) { m.Inputs[0].Shape = []int64{-1, 3, 1} },
		"a fixed dimension":     func(m *onnx.Manifest) { m.Inputs[0].Shape = []int64{-1, 4} },
		"an output name":        func(m *onnx.Manifest) { m.Outputs[0].Name = "scores" },
		"an output type":        func(m *onnx.Manifest) { m.Outputs[1].ElementType = onnx.Float32 },
	} {
		t.Run(name, func(t *testing.T) {
			manifest, dir := writeModel(t, affineManifest(), affineModel())
			change(&manifest)
			if _, err := runtime.Open(manifest, dir); !errors.Is(err, onnx.ErrContractMismatch) {
				t.Fatalf("Open = %v, want ErrContractMismatch", err)
			}
		})
	}
}

func TestInputsOutsideTheContractAreRefusedBeforeTheRun(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}

	for name, inputs := range map[string]map[string]onnx.Tensor{
		"no input":             {},
		"an unknown input":     {"x": {Shape: []int64{1, 3}, Float32: make([]float32, 3)}, "z": {Shape: []int64{1}, Float32: []float32{0}}},
		"the wrong type":       {"x": {Shape: []int64{1, 3}, Int64: make([]int64, 3)}},
		"the wrong width":      {"x": {Shape: []int64{1, 4}, Float32: make([]float32, 4)}},
		"too little data":      {"x": {Shape: []int64{2, 3}, Float32: make([]float32, 5)}},
		"two data slices":      {"x": {Shape: []int64{1, 3}, Float32: make([]float32, 3), Uint8: make([]uint8, 3)}},
		"a zero dimension":     {"x": {Shape: []int64{0, 3}, Float32: []float32{}}},
		"the wrong rank":       {"x": {Shape: []int64{3}, Float32: make([]float32, 3)}},
		"a negative dimension": {"x": {Shape: []int64{-1, 3}, Float32: make([]float32, 3)}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := session.Run(context.Background(), inputs); !errors.Is(err, onnx.ErrInvalidTensor) {
				t.Fatalf("Run = %v, want ErrInvalidTensor", err)
			}
		})
	}
}

func TestADoneContextIsRefusedBeforeTheRun(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = session.Run(ctx, map[string]onnx.Tensor{
		"x": {Shape: []int64{1, 3}, Float32: make([]float32, 3)},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v, want context.Canceled", err)
	}
}

// TestConcurrentRunsAgreeWithSerialOnes is the one that matters under -race:
// many goroutines share one session, and each must get its own answer.
func TestConcurrentRunsAgreeWithSerialOnes(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{IntraOpThreads: 1})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}

	const workers, runs = 8, 25
	var wg sync.WaitGroup
	errs := make(chan error, workers*runs)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < runs; i++ {
				v := float32(w*runs + i)
				result, err := session.Run(context.Background(), map[string]onnx.Tensor{
					"x": {Shape: []int64{1, 3}, Float32: []float32{v, 0, 1}},
				})
				if err != nil {
					errs <- err
					return
				}
				// y = {v + 5.5, 2v + 5.5}.
				y := result.Outputs["y"].Float32
				if y[0] != v+5.5 || y[1] != 2*v+5.5 {
					errs <- errors.New("a run returned another run's answer")
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestClosingASessionWaitsForItsRunsAndRefusesNewOnes(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				_, err := session.Run(context.Background(), map[string]onnx.Tensor{
					"x": {Shape: []int64{1, 3}, Float32: make([]float32, 3)},
				})
				if err != nil && !errors.Is(err, onnx.ErrClosed) {
					t.Errorf("a run failed for another reason than the close: %v", err)
				}
			}
		}()
	}
	if err := session.Close(); err != nil {
		t.Fatalf("closing the session: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("closing it twice: %v", err)
	}
	wg.Wait()

	_, err = session.Run(context.Background(), map[string]onnx.Tensor{
		"x": {Shape: []int64{1, 3}, Float32: make([]float32, 3)},
	})
	if !errors.Is(err, onnx.ErrClosed) {
		t.Fatalf("Run after Close = %v, want ErrClosed", err)
	}
}

func TestInitAndCloseAreIdempotentAndSafeTogether(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening the model: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := runtime.Init(); err != nil && !errors.Is(err, onnx.ErrClosed) {
				t.Errorf("Init: %v", err)
			}
		}()
	}
	wg.Wait()
	if err := runtime.Health(); err != nil {
		t.Fatalf("Health after Init = %v", err)
	}

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := runtime.Close(); err != nil {
				t.Errorf("Close: %v", err)
			}
		}()
	}
	wg.Wait()

	if err := runtime.Health(); !errors.Is(err, onnx.ErrClosed) {
		t.Errorf("Health after Close = %v, want ErrClosed", err)
	}
	if err := runtime.Init(); !errors.Is(err, onnx.ErrClosed) {
		t.Errorf("Init after Close = %v, want ErrClosed", err)
	}
	if _, err := runtime.Open(manifest, dir); !errors.Is(err, onnx.ErrClosed) {
		t.Errorf("Open after Close = %v, want ErrClosed", err)
	}
	if _, err := session.Run(context.Background(), map[string]onnx.Tensor{
		"x": {Shape: []int64{1, 3}, Float32: make([]float32, 3)},
	}); !errors.Is(err, onnx.ErrClosed) {
		t.Errorf("a session of a closed runtime ran: %v", err)
	}
}

func TestTheLibraryIsLoadedAgainAfterTheLastRuntimeCloses(t *testing.T) {
	first := newRuntime(t, onnx.Config{})
	if err := first.Close(); err != nil {
		t.Fatalf("closing: %v", err)
	}

	second := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := second.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening on a second runtime: %v", err)
	}
	if _, err := session.Run(context.Background(), map[string]onnx.Tensor{
		"x": {Shape: []int64{1, 3}, Float32: make([]float32, 3)},
	}); err != nil {
		t.Fatalf("running on a second runtime: %v", err)
	}
}

func TestAnUnusedRuntimeRefusesToOpen(t *testing.T) {
	t.Parallel()

	runtime, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatal(err)
	}
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	if _, err := runtime.Open(manifest, dir); !errors.Is(err, onnx.ErrNotInitialized) {
		t.Fatalf("Open before Init = %v, want ErrNotInitialized", err)
	}
	if err := runtime.Health(); !errors.Is(err, onnx.ErrNotInitialized) {
		t.Fatalf("Health before Init = %v, want ErrNotInitialized", err)
	}
	if err := runtime.Close(); err != nil {
		t.Fatalf("closing a runtime that never loaded anything: %v", err)
	}
}

// TestAMissingLibraryIsReportedAsUnavailable is not parallel: a process loads
// one library, and a runtime alive in another test would make this one's
// refusal be about that instead.
func TestAMissingLibraryIsReportedAsUnavailable(t *testing.T) {
	runtime, err := onnx.NewRuntime(onnx.Config{SharedLibraryPath: filepath.Join(t.TempDir(), "libonnxruntime.dylib")})
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Init(); !errors.Is(err, onnx.ErrLibraryUnavailable) {
		t.Fatalf("Init = %v, want ErrLibraryUnavailable", err)
	}
	if err := runtime.Health(); !errors.Is(err, onnx.ErrNotInitialized) {
		t.Fatalf("a failed Init left the runtime in another state: %v", err)
	}
}

func TestAnAcceleratorTheLibraryLacksFallsBackToTheCPU(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{Providers: []string{onnx.ProviderCUDA, onnx.ProviderCoreML, onnx.ProviderCPU}})

	statuses := runtime.Providers()
	if len(statuses) != 3 {
		t.Fatalf("Providers() = %+v, want one status per configured provider", statuses)
	}
	for _, status := range statuses {
		if !status.Available && status.Reason == "" {
			t.Errorf("%s is unavailable and no reason is given", status.Name)
		}
		t.Logf("provider %s available=%v %s", status.Name, status.Available, status.Reason)
	}
	if last := statuses[2]; last.Name != onnx.ProviderCPU || !last.Available {
		t.Errorf("the CPU is %+v, want available", last)
	}

	manifest, dir := writeModel(t, affineManifest(), affineModel())
	session, err := runtime.Open(manifest, dir)
	if err != nil {
		t.Fatalf("opening with accelerators preferred: %v", err)
	}
	result, err := session.Run(context.Background(), map[string]onnx.Tensor{
		"x": {Shape: []int64{1, 3}, Float32: []float32{1, 0, 0}},
	})
	if err != nil {
		t.Fatalf("running with accelerators preferred: %v", err)
	}
	assertFloats(t, "y", result.Outputs["y"].Float32, 1.5, 1.5)
	t.Logf("the session was built for %s", result.Model.Provider)

	// The provider is the first one that was available, or the CPU.
	want := onnx.ProviderCPU
	for _, status := range statuses {
		if status.Available {
			want = status.Name
			break
		}
	}
	if got := result.Model.Provider; got != want && got != onnx.ProviderCPU {
		t.Errorf("Provider = %q, want %q or the CPU fallback", got, want)
	}
}

func TestTheConfigurationIsCheckedWhereItIsWritten(t *testing.T) {
	t.Parallel()

	for name, cfg := range map[string]onnx.Config{
		"an unknown provider":  {Providers: []string{"tpu"}},
		"a provider twice":     {Providers: []string{"cpu", "cpu"}},
		"negative threads":     {IntraOpThreads: -1},
		"implausible threads":  {IntraOpThreads: onnx.MaxIntraOpThreads + 1},
		"an uppercase name":    {Providers: []string{"CPU"}},
		"an empty string name": {Providers: []string{""}},
	} {
		if _, err := onnx.NewRuntime(cfg); err == nil {
			t.Errorf("%s: NewRuntime accepted %+v", name, cfg)
		}
		if _, err := onnx.New(cfg); err == nil {
			t.Errorf("%s: New accepted %+v", name, cfg)
		}
	}
}

func assertShape(t *testing.T, name string, got []int64, want ...int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s shape = %v, want %v", name, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s shape = %v, want %v", name, got, want)
		}
	}
}

func assertFloats(t *testing.T, name string, got []float32, want ...float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}
}
