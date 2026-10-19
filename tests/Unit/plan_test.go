package unit_test

import (
	"maps"
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
)

// The plan is what a session is built with. These tests run on any machine:
// they check the options handed to ONNX Runtime, not that a GPU takes them.

const (
	modelDigest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
	tableDigest = "60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752"
	cacheDir    = "/var/cache/kuaa/tensorrt"
)

func floatManifest(t *testing.T) onnx.Manifest {
	t.Helper()
	m, err := onnx.LoadManifest([]byte(manifestJSON))
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	return m
}

func calibratedManifest(t *testing.T, format calibration.Format) onnx.Manifest {
	t.Helper()
	m := floatManifest(t)
	m.Calibration = &onnx.Calibration{
		File:      "det/calibration.table",
		SHA256:    tableDigest,
		Format:    format,
		Precision: onnx.PrecisionINT8,
		Method:    onnx.CalibrationMinMax,
		Source:    "arandu-onnx-calibrate over 200 synthetic samples",
	}
	return m
}

func gpuConfig() onnx.Config {
	off := false
	return onnx.Config{
		Providers: []string{onnx.ProviderTensorRT, onnx.ProviderCUDA, onnx.ProviderCPU},
		Device:    1,
		CUDA: onnx.CUDAOptions{
			GPUMemLimit:         4 << 30,
			ConvAlgoSearch:      "HEURISTIC",
			ConvUseMaxWorkspace: &off,
			ArenaExtendStrategy: "kSameAsRequested",
			Extra:               map[string]string{"do_copy_in_default_stream": "1"},
		},
		TensorRT: onnx.TensorRTOptions{
			CacheDir:                 cacheDir,
			EngineCache:              true,
			TimingCache:              true,
			FP16:                     true,
			INT8:                     true,
			MaxWorkspaceSize:         2 << 30,
			BuilderOptimizationLevel: 4,
			Extra:                    map[string]string{"trt_profile_min_shapes": "x:1x3x640x640"},
		},
	}
}

func plan(t *testing.T, cfg onnx.Config, m onnx.Manifest, device int) onnx.SessionPlan {
	t.Helper()
	p, err := cfg.Plan(m, device)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	return p
}

func TestAConfigurationWithoutAcceleratorsPlansNoProviderOptions(t *testing.T) {
	t.Parallel()

	p := plan(t, onnx.Config{}, floatManifest(t), 0)
	if len(p.Accelerators) != 0 || p.CUDA != nil || p.TensorRT != nil || p.INT8 || p.EngineCacheDir != "" {
		t.Errorf("Plan of the zero Config = %+v, want nothing but the CPU", p)
	}

	// A v0.1.0 configuration naming CUDA gets the device and nothing else,
	// which is ONNX Runtime's default: what it ran with before.
	p = plan(t, onnx.Config{Providers: []string{onnx.ProviderCoreML, onnx.ProviderCUDA}}, floatManifest(t), 0)
	if !maps.Equal(p.CUDA, map[string]string{"device_id": "0"}) {
		t.Errorf("CUDA options = %v, want only device_id 0", p.CUDA)
	}
	if strings.Join(p.Accelerators, ",") != "coreml,cuda" {
		t.Errorf("Accelerators = %v", p.Accelerators)
	}
}

func TestTheTensorRTPlanOfACalibratedFloatModel(t *testing.T) {
	t.Parallel()

	m := calibratedManifest(t, calibration.FormatONNXRuntime)
	p := plan(t, gpuConfig(), m, 1)

	dir := cacheDir + "/paddleocr.det-9f86d081884c7d65-60303ae22b998861"
	want := map[string]string{
		"device_id":                             "1",
		"trt_fp16_enable":                       "1",
		"trt_int8_enable":                       "1",
		"trt_int8_calibration_table_name":       "calibration.flatbuffers",
		"trt_int8_use_native_calibration_table": "0",
		"trt_engine_cache_enable":               "1",
		"trt_engine_cache_path":                 dir,
		"trt_timing_cache_enable":               "1",
		"trt_timing_cache_path":                 cacheDir,
		"trt_max_workspace_size":                "2147483648",
		"trt_builder_optimization_level":        "4",
		"trt_profile_min_shapes":                "x:1x3x640x640",
	}
	if !maps.Equal(p.TensorRT, want) {
		t.Errorf("TensorRT options =\n%v\nwant\n%v", p.TensorRT, want)
	}
	wantCUDA := map[string]string{
		"device_id":                    "1",
		"gpu_mem_limit":                "4294967296",
		"cudnn_conv_algo_search":       "HEURISTIC",
		"cudnn_conv_use_max_workspace": "0",
		"arena_extend_strategy":        "kSameAsRequested",
		"do_copy_in_default_stream":    "1",
	}
	if !maps.Equal(p.CUDA, wantCUDA) {
		t.Errorf("CUDA options =\n%v\nwant\n%v", p.CUDA, wantCUDA)
	}
	if !p.INT8 || p.Calibration != tableDigest || p.CalibrationFormat != calibration.FormatONNXRuntime ||
		p.EngineCacheDir != dir || p.CalibrationPath != dir+"/calibration.flatbuffers" {
		t.Errorf("Plan = %+v", p)
	}
	if strings.Join(p.Accelerators, ",") != "tensorrt,cuda" {
		t.Errorf("Accelerators = %v, want tensorrt before cuda", p.Accelerators)
	}
}

func TestATensorRTFormatTableIsReadAsNative(t *testing.T) {
	t.Parallel()

	p := plan(t, gpuConfig(), calibratedManifest(t, calibration.FormatTensorRT), 0)
	if p.TensorRT["trt_int8_use_native_calibration_table"] != "1" ||
		p.TensorRT["trt_int8_calibration_table_name"] != "calibration.cache" {
		t.Errorf("TensorRT options = %v, want the native table calibration.cache", p.TensorRT)
	}
}

func TestINT8NeedsSomethingToTakeTheRangesFrom(t *testing.T) {
	t.Parallel()

	// A float model with no table runs without INT8 whatever the config says.
	p := plan(t, gpuConfig(), floatManifest(t), 0)
	if p.INT8 || p.TensorRT["trt_int8_enable"] != "0" || p.Calibration != "" {
		t.Errorf("a float model without a table: %+v", p)
	}
	if _, set := p.TensorRT["trt_int8_calibration_table_name"]; set {
		t.Errorf("a float model without a table was given a table name: %v", p.TensorRT)
	}

	// A QDQ model carries its scales and takes INT8 with no table.
	qdq := floatManifest(t)
	qdq.Precision = onnx.PrecisionINT8
	p = plan(t, gpuConfig(), qdq, 0)
	if !p.INT8 || p.TensorRT["trt_int8_enable"] != "1" || p.CalibrationPath != "" {
		t.Errorf("a QDQ model: %+v", p)
	}

	// INT8 off in the config leaves the table unused.
	cfg := gpuConfig()
	cfg.TensorRT.INT8 = false
	p = plan(t, cfg, calibratedManifest(t, calibration.FormatONNXRuntime), 0)
	if p.INT8 || p.Calibration != "" || p.TensorRT["trt_int8_enable"] != "0" {
		t.Errorf("INT8 off: %+v", p)
	}
}

func TestEveryModelVersionAndTableGetsItsOwnEngineCache(t *testing.T) {
	t.Parallel()

	// ONNX Runtime keys an engine by tensor names, not weights, so a new
	// version of the same architecture would load the old engine from a
	// shared directory.
	a := calibratedManifest(t, calibration.FormatONNXRuntime)
	b := calibratedManifest(t, calibration.FormatONNXRuntime)
	b.SHA256 = strings.Repeat("1", 64)
	c := calibratedManifest(t, calibration.FormatONNXRuntime)
	c.Calibration.SHA256 = strings.Repeat("2", 64)

	dirs := map[string]bool{}
	for _, m := range []onnx.Manifest{a, b, c} {
		dirs[plan(t, gpuConfig(), m, 0).EngineCacheDir] = true
	}
	if len(dirs) != 3 {
		t.Errorf("engine cache directories = %v, want one per model digest and table digest", dirs)
	}
}

func TestAProviderConfigurationThatCannotWorkIsRefused(t *testing.T) {
	t.Parallel()

	for name, change := range map[string]func(*onnx.Config){
		"unknown provider":  func(c *onnx.Config) { c.Providers = []string{"rocm"} },
		"device":            func(c *onnx.Config) { c.Device = onnx.MaxDevice + 1 },
		"negative memory":   func(c *onnx.Config) { c.CUDA.GPUMemLimit = -1 },
		"conv search":       func(c *onnx.Config) { c.CUDA.ConvAlgoSearch = "FAST" },
		"arena":             func(c *onnx.Config) { c.CUDA.ArenaExtendStrategy = "grow" },
		"cuda extra device": func(c *onnx.Config) { c.CUDA.Extra = map[string]string{"device_id": "3"} },
		"relative cache":    func(c *onnx.Config) { c.TensorRT.CacheDir = "cache/trt" },
		"unclean cache":     func(c *onnx.Config) { c.TensorRT.CacheDir = "/var/cache/../trt" },
		"int8 without cache": func(c *onnx.Config) {
			c.TensorRT.CacheDir = ""
			c.TensorRT.EngineCache = false
			c.TensorRT.TimingCache = false
		},
		"engine cache, no dir": func(c *onnx.Config) { c.TensorRT = onnx.TensorRTOptions{EngineCache: true} },
		"builder level":        func(c *onnx.Config) { c.TensorRT.BuilderOptimizationLevel = 6 },
		"workspace":            func(c *onnx.Config) { c.TensorRT.MaxWorkspaceSize = -1 },
		"managed trt key":      func(c *onnx.Config) { c.TensorRT.Extra = map[string]string{"trt_engine_cache_path": "/tmp"} },
		"plugin library":       func(c *onnx.Config) { c.TensorRT.Extra = map[string]string{"trt_extra_plugin_lib_paths": "/tmp/x.so"} },
		"decryption library": func(c *onnx.Config) {
			c.TensorRT.Extra = map[string]string{"trt_engine_decryption_lib_path": "/tmp/x.so"}
		},
		"raw pointer":              func(c *onnx.Config) { c.TensorRT.Extra = map[string]string{"trt_onnx_bytestream": "0x1234"} },
		"not a trt key":            func(c *onnx.Config) { c.TensorRT.Extra = map[string]string{"device": "0"} },
		"require with no provider": func(c *onnx.Config) { *c = onnx.Config{Placement: onnx.PlacementOptions{RequireAccelerated: true}} },
		"empty cpu op":             func(c *onnx.Config) { c.Placement.CPUOps = []string{" "} },
	} {
		cfg := gpuConfig()
		change(&cfg)
		if err := cfg.Validate(); err == nil {
			t.Errorf("%s: Validate accepted %+v", name, cfg)
		}
		if _, err := onnx.New(cfg); err == nil {
			t.Errorf("%s: New accepted the configuration", name)
		}
	}
	if err := gpuConfig().Validate(); err != nil {
		t.Errorf("the GPU configuration is refused: %v", err)
	}
}

func TestTheRuntimeKeepsItsOwnCopyOfTheConfiguration(t *testing.T) {
	t.Parallel()

	cfg := gpuConfig()
	runtime, err := onnx.NewRuntime(cfg)
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	cfg.TensorRT.Extra["trt_profile_min_shapes"] = "changed"
	*cfg.CUDA.ConvUseMaxWorkspace = true
	got := runtime.Config()
	if got.TensorRT.Extra["trt_profile_min_shapes"] != "x:1x3x640x640" || *got.CUDA.ConvUseMaxWorkspace {
		t.Errorf("the runtime's configuration changed with the caller's: %+v", got)
	}
	got.CUDA.Extra["do_copy_in_default_stream"] = "0"
	if runtime.Config().CUDA.Extra["do_copy_in_default_stream"] != "1" {
		t.Error("changing what Config returned changed the runtime")
	}
	if len(got.Placement.CPUOps) == 0 {
		t.Error("Config().Placement.CPUOps is empty, want DefaultCPUOps filled in")
	}
}
