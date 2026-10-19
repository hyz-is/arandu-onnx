package onnx

import (
	"fmt"
	"maps"
	"path/filepath"
	"runtime"
	"strings"
)

// The execution providers a Config may name.
const (
	// ProviderCPU is ONNX Runtime's own kernels. It is always available, and it
	// is where every node another provider does not take runs.
	ProviderCPU = "cpu"
	// ProviderCoreML is Apple's Core ML, in builds of ONNX Runtime for Apple
	// platforms that include it.
	ProviderCoreML = "coreml"
	// ProviderCUDA is NVIDIA CUDA, in builds of ONNX Runtime that include it and
	// on a machine with a driver and a card.
	ProviderCUDA = "cuda"
	// ProviderTensorRT is NVIDIA TensorRT, in GPU builds of ONNX Runtime on a
	// machine with the TensorRT libraries. It compiles the subgraphs it
	// supports into engines, and is the provider that runs INT8 on the tensor
	// cores of a Turing card such as the Tesla T4. It is named before
	// ProviderCUDA, which then takes the nodes TensorRT leaves.
	ProviderTensorRT = "tensorrt"
)

// MaxIntraOpThreads is the ceiling Config.IntraOpThreads is refused above. A
// number past any machine's core count is a typo, not a tuning decision.
const MaxIntraOpThreads = 1024

// MaxDevice is the highest GPU device id a Config or OpenOn accepts.
const MaxDevice = 255

// DefaultSharedLibraryPath is the ONNX Runtime library Config uses when
// SharedLibraryPath is empty, for the operating system and architecture the
// binary was built for.
//
// On macOS it is the Homebrew location. On Linux it is the bare file name,
// which the dynamic loader resolves through its own search path. On Windows it
// is the DLL name, resolved the same way.
func DefaultSharedLibraryPath() string {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "/opt/homebrew/lib/libonnxruntime.dylib"
		}
		return "/usr/local/lib/libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "libonnxruntime.so"
	}
}

// Config is what the application passes when it wires this package.
//
// Every value comes from the application's own typed configuration. This
// package reads no environment variable: a setting that could arrive from two
// places is a setting nobody can say the value of.
type Config struct {
	// SharedLibraryPath is the ONNX Runtime shared library to load. Empty means
	// DefaultSharedLibraryPath().
	SharedLibraryPath string

	// IntraOpThreads is how many threads one inference may use inside an
	// operator. Zero leaves the choice to ONNX Runtime, which uses the physical
	// cores.
	IntraOpThreads int

	// Providers is the execution providers to try, in order of preference:
	// ProviderCPU, ProviderCoreML, ProviderCUDA, ProviderTensorRT. Empty means
	// ProviderCPU alone. On an NVIDIA machine the order that uses everything
	// is tensorrt, cuda, cpu.
	//
	// A provider the loaded library or the machine cannot offer is skipped and
	// reported by Runtime.Providers, not refused: the CPU is always the last
	// resort, so naming an accelerator is a preference and never a reason for
	// the application not to start. When a session cannot be built with every
	// available accelerator, the first is dropped and the build is tried again
	// with the rest, down to the CPU alone.
	Providers []string

	// Device is the GPU that ProviderCUDA and ProviderTensorRT sessions opened
	// with Open run on. Runtime.OpenOn opens a session on another one, which
	// is how a model is spread over the cards of a machine.
	Device int

	// CUDA tunes ProviderCUDA. The zero value is ONNX Runtime's defaults.
	CUDA CUDAOptions

	// TensorRT tunes ProviderTensorRT. The zero value is ONNX Runtime's
	// defaults, with INT8 and every cache off.
	TensorRT TensorRTOptions

	// Placement asks for the report of which provider each node of a model
	// landed on, and for the refusal of a session whose work fell back to the
	// CPU.
	Placement PlacementOptions
}

// CUDAOptions is the ProviderCUDA options this package sets. A zero field
// leaves ONNX Runtime's default.
type CUDAOptions struct {
	// GPUMemLimit caps the provider's memory arena for one session, in bytes.
	// Zero is unlimited. With several sessions on one card, an unlimited arena
	// lets the first large allocation take what the others will need.
	GPUMemLimit int64

	// ConvAlgoSearch is how cuDNN picks a convolution algorithm: "EXHAUSTIVE"
	// (ONNX Runtime's default, the fastest steady state and the slowest
	// session build), "HEURISTIC" or "DEFAULT". Concurrent sessions on one
	// card have been reported to conflict with the exhaustive search
	// (onnxruntime issue 26610).
	ConvAlgoSearch string

	// ConvUseMaxWorkspace lets cuDNN allocate the workspace it asks for, which
	// is what lets an FP16 model reach the tensor cores. Nil leaves ONNX
	// Runtime's default, which is on.
	ConvUseMaxWorkspace *bool

	// ArenaExtendStrategy is how the arena grows: "kNextPowerOfTwo" (the
	// default) or "kSameAsRequested", which keeps several sessions on one card
	// from each reserving a power of two.
	ArenaExtendStrategy string

	// PreferNHWC asks for NHWC kernels, transposing the graph where needed. It
	// can help on tensor cores or cost more in transposes than it saves; it is
	// a setting to measure.
	PreferNHWC bool

	// Extra is any other CUDA provider option, by ONNX Runtime's name. A key
	// one of the fields above sets, or device_id, is refused here: it has one
	// place to be written.
	Extra map[string]string
}

// TensorRTOptions is the ProviderTensorRT options this package sets.
//
// INT8 has two forms, and both are switched on by INT8. A model whose
// manifest declares precision int8 is a QDQ export: TensorRT reads the scales
// from the graph and needs no table. A float model whose manifest carries a
// Calibration is implicit quantization: TensorRT is handed the verified table
// and runs in INT8 every tensor it names, and in FP16 (with FP16) or FP32 the
// rest. A float model with neither runs without INT8 whatever this says.
type TensorRTOptions struct {
	// CacheDir is an absolute directory, private to the application, that
	// holds the compiled engines, the timing cache and the verified copy of
	// each calibration table. Each model gets a subdirectory named by its id
	// and the digests of its file and its table, because ONNX Runtime keys an
	// engine by tensor names and not by weights: two versions of one
	// architecture would otherwise load each other's engines. Building the
	// engines of a large model takes minutes to tens of minutes, so the
	// directory belongs on a volume that survives a restart. Required by
	// EngineCache, TimingCache and INT8.
	CacheDir string

	// EngineCache keeps compiled engines in CacheDir and loads them on the
	// next start instead of building them again. An engine is specific to the
	// card's architecture and to the TensorRT version.
	EngineCache bool

	// TimingCache keeps TensorRT's layer timing measurements in CacheDir,
	// shared by every model, which shortens the next build.
	TimingCache bool

	// FP16 lets TensorRT run in half precision what INT8 does not take.
	FP16 bool

	// INT8 lets TensorRT run in INT8 the models that can be: see the type's
	// comment.
	INT8 bool

	// MaxWorkspaceSize is the memory TensorRT may use to build and run an
	// engine, in bytes. Zero is ONNX Runtime's default, 1 GiB.
	MaxWorkspaceSize int64

	// BuilderOptimizationLevel is TensorRT's builder optimization level, 1 to
	// 5, higher building slower for a faster engine. Zero is ONNX Runtime's
	// default, 3.
	BuilderOptimizationLevel int

	// Extra is any other TensorRT provider option, by ONNX Runtime's name,
	// such as trt_profile_min_shapes for a model with dynamic dimensions. A
	// key the fields above set is refused, and so are the options that make
	// ONNX Runtime load a library, write files elsewhere or read a raw
	// pointer: trt_engine_decryption_enable, trt_engine_decryption_lib_path,
	// trt_extra_plugin_lib_paths, trt_dump_subgraphs, trt_dump_ep_context_model,
	// trt_ep_context_file_path, trt_onnx_bytestream and
	// trt_external_data_bytestream, with their sizes.
	Extra map[string]string
}

// PlacementOptions is the node placement report and its guard.
//
// ONNX Runtime places each node of a graph on the first provider that takes
// it, and a node no accelerator takes runs on the CPU without an error: the
// model only gets slower. The only account of where nodes went is the log ONNX
// Runtime writes at verbose severity while it builds a session, so the first
// session of each model on each device is built at that severity with the
// process's standard error captured, parsed, and written back.
type PlacementOptions struct {
	// Capture builds the first session of each model on each device, for each
	// list of providers it is built with, with the placement log captured,
	// and reports it through Session.Placement. A CPU session is captured too,
	// which is what lets a machine without a GPU show the capture works.
	//
	// The capture redirects the process's file descriptor 2 for the length of
	// one session build, so it is serialised, and what anything else writes to
	// standard error meanwhile is written back afterwards. It is supported on
	// Linux and the BSDs, macOS included; elsewhere the report says it was not
	// captured.
	Capture bool

	// RequireAccelerated refuses, with ErrCPUFallback, a session whose real
	// work is not on an accelerator: one that fell back to the CPU alone, one
	// that placed on the CPU a node whose operator is not in CPUOps, one where
	// the provider asked for first got no node at all, and one whose placement
	// could not be captured, because what was not measured is not taken for
	// a pass. It implies Capture.
	RequireAccelerated bool

	// CPUOps is the operators an accelerated session may place on the CPU
	// without it counting as a fallback. Nil means DefaultCPUOps: the shape
	// arithmetic ONNX Runtime keeps on the CPU on purpose.
	CPUOps []string
}

// DefaultCPUOps is the operators PlacementOptions accepts on the CPU when
// CPUOps is nil.
//
// ONNX Runtime keeps on the CPU the small int64 arithmetic that turns one
// shape into another, because a round trip to the card for four numbers costs
// more than the arithmetic; its own warning says "ORT explicitly assigns shape
// related ops to CPU to improve perf". The list is arandu-alpr's, measured on
// detector and OCR graphs. It includes elementwise operators such as Add and
// Mul because shape arithmetic uses them, so an elementwise operator on real
// data that fell to the CPU is not caught by it; a deployment that wants that
// caught names a shorter list.
func DefaultCPUOps() []string {
	return []string{
		"Shape", "Size", "Gather", "Unsqueeze", "Squeeze", "Concat", "Slice", "Cast",
		"ConstantOfShape", "Range", "Reshape", "Expand", "Tile", "Equal", "Where", "Not",
		"Less", "Greater", "Add", "Sub", "Mul", "Div", "Floor", "Ceil", "Min", "Max",
		"ReduceProd", "Identity", "Constant",
	}
}

// managedCUDAOptions is the CUDA option names a CUDAOptions field or the
// device sets.
var managedCUDAOptions = map[string]bool{
	"device_id":                    true,
	"gpu_mem_limit":                true,
	"cudnn_conv_algo_search":       true,
	"cudnn_conv_use_max_workspace": true,
	"arena_extend_strategy":        true,
	"prefer_nhwc":                  true,
}

// managedTensorRTOptions is the TensorRT option names a TensorRTOptions field,
// the device or the manifest sets.
var managedTensorRTOptions = map[string]bool{
	"device_id":                             true,
	"trt_max_workspace_size":                true,
	"trt_fp16_enable":                       true,
	"trt_int8_enable":                       true,
	"trt_int8_calibration_table_name":       true,
	"trt_int8_use_native_calibration_table": true,
	"trt_engine_cache_enable":               true,
	"trt_engine_cache_path":                 true,
	"trt_timing_cache_enable":               true,
	"trt_timing_cache_path":                 true,
	"trt_builder_optimization_level":        true,
}

// refusedTensorRTOptions is the TensorRT option names that would make ONNX
// Runtime load a library, write outside CacheDir, or read memory at an address
// given as text.
var refusedTensorRTOptions = map[string]bool{
	"trt_engine_decryption_enable":      true,
	"trt_engine_decryption_lib_path":    true,
	"trt_extra_plugin_lib_paths":        true,
	"trt_dump_subgraphs":                true,
	"trt_dump_ep_context_model":         true,
	"trt_ep_context_file_path":          true,
	"trt_onnx_bytestream":               true,
	"trt_onnx_bytestream_size":          true,
	"trt_external_data_bytestream":      true,
	"trt_external_data_bytestream_size": true,
}

// Validate reports what the configuration cannot be used with.
//
// It is called by NewRuntime and New, so a setting that cannot work fails
// where it is wired rather than on the first inference.
func (c Config) Validate() error {
	if c.IntraOpThreads < 0 || c.IntraOpThreads > MaxIntraOpThreads {
		return fmt.Errorf("onnx: Config.IntraOpThreads is %d, and has to be between 0 and %d, where 0 lets ONNX Runtime choose",
			c.IntraOpThreads, MaxIntraOpThreads)
	}
	seen := map[string]bool{}
	accelerated := false
	for _, provider := range c.Providers {
		switch provider {
		case ProviderCPU:
		case ProviderCoreML, ProviderCUDA, ProviderTensorRT:
			accelerated = true
		default:
			return fmt.Errorf("onnx: Config.Providers names %q, which is not one of %q, %q, %q or %q",
				provider, ProviderCPU, ProviderCoreML, ProviderCUDA, ProviderTensorRT)
		}
		if seen[provider] {
			return fmt.Errorf("onnx: Config.Providers names %q twice", provider)
		}
		seen[provider] = true
	}
	if err := validateDevice("Config.Device", c.Device); err != nil {
		return err
	}
	if err := c.CUDA.validate(); err != nil {
		return err
	}
	if err := c.TensorRT.validate(); err != nil {
		return err
	}
	if c.Placement.RequireAccelerated && !accelerated {
		return fmt.Errorf("onnx: Config.Placement.RequireAccelerated is set and Config.Providers names no accelerator, so every session would be refused")
	}
	for _, op := range c.Placement.CPUOps {
		if strings.TrimSpace(op) == "" {
			return fmt.Errorf("onnx: Config.Placement.CPUOps has an empty operator name")
		}
	}
	return nil
}

// validateDevice checks a GPU device id.
func validateDevice(what string, device int) error {
	if device < 0 || device > MaxDevice {
		return fmt.Errorf("onnx: %s is %d, and a GPU device id is between 0 and %d", what, device, MaxDevice)
	}
	return nil
}

func (o CUDAOptions) validate() error {
	if o.GPUMemLimit < 0 {
		return fmt.Errorf("onnx: Config.CUDA.GPUMemLimit is %d, and has to be 0 (unlimited) or a number of bytes", o.GPUMemLimit)
	}
	switch o.ConvAlgoSearch {
	case "", "EXHAUSTIVE", "HEURISTIC", "DEFAULT":
	default:
		return fmt.Errorf("onnx: Config.CUDA.ConvAlgoSearch is %q, and has to be EXHAUSTIVE, HEURISTIC or DEFAULT", o.ConvAlgoSearch)
	}
	switch o.ArenaExtendStrategy {
	case "", "kNextPowerOfTwo", "kSameAsRequested":
	default:
		return fmt.Errorf("onnx: Config.CUDA.ArenaExtendStrategy is %q, and has to be kNextPowerOfTwo or kSameAsRequested", o.ArenaExtendStrategy)
	}
	for key := range o.Extra {
		if strings.TrimSpace(key) == "" {
			return fmt.Errorf("onnx: Config.CUDA.Extra has an empty key")
		}
		if managedCUDAOptions[key] {
			return fmt.Errorf("onnx: Config.CUDA.Extra sets %q, which a CUDAOptions field or the device sets", key)
		}
	}
	return nil
}

func (o TensorRTOptions) validate() error {
	if o.CacheDir != "" && (!filepath.IsAbs(o.CacheDir) || filepath.Clean(o.CacheDir) != o.CacheDir) {
		return fmt.Errorf("onnx: Config.TensorRT.CacheDir is %q, and has to be a clean absolute path: a relative one would move with the working directory", o.CacheDir)
	}
	for _, needs := range []struct {
		set  bool
		name string
	}{{o.EngineCache, "EngineCache"}, {o.TimingCache, "TimingCache"}, {o.INT8, "INT8"}} {
		if needs.set && o.CacheDir == "" {
			return fmt.Errorf("onnx: Config.TensorRT.%s needs Config.TensorRT.CacheDir, where the engines, the timing cache and the verified calibration tables are kept", needs.name)
		}
	}
	if o.MaxWorkspaceSize < 0 {
		return fmt.Errorf("onnx: Config.TensorRT.MaxWorkspaceSize is %d, and has to be 0 (the default) or a number of bytes", o.MaxWorkspaceSize)
	}
	if o.BuilderOptimizationLevel < 0 || o.BuilderOptimizationLevel > 5 {
		return fmt.Errorf("onnx: Config.TensorRT.BuilderOptimizationLevel is %d, and has to be 0 (the default) or 1 to 5", o.BuilderOptimizationLevel)
	}
	for key := range o.Extra {
		switch {
		case !strings.HasPrefix(key, "trt_"):
			return fmt.Errorf("onnx: Config.TensorRT.Extra sets %q, and TensorRT provider options start with trt_", key)
		case managedTensorRTOptions[key]:
			return fmt.Errorf("onnx: Config.TensorRT.Extra sets %q, which a TensorRTOptions field, the device or the manifest sets", key)
		case refusedTensorRTOptions[key]:
			return fmt.Errorf("onnx: Config.TensorRT.Extra sets %q, which makes ONNX Runtime load a library, write outside CacheDir or read a raw pointer, and is refused", key)
		}
	}
	return nil
}

// withDefaults returns the configuration with the optional fields filled in,
// and with its own copy of every slice, map and pointer.
//
// It runs after Validate and never before: filling a default in first would
// hide the value somebody actually wrote from the check that would have refused
// it.
func (c Config) withDefaults() Config {
	if c.SharedLibraryPath == "" {
		c.SharedLibraryPath = DefaultSharedLibraryPath()
	}
	if len(c.Providers) == 0 {
		c.Providers = []string{ProviderCPU}
	}
	if c.Placement.RequireAccelerated {
		c.Placement.Capture = true
	}
	if c.Placement.CPUOps == nil {
		c.Placement.CPUOps = DefaultCPUOps()
	}
	return c.clone()
}

// clone copies every slice, map and pointer, so a caller changing what it
// passed or what it got back does not change the runtime.
func (c Config) clone() Config {
	c.Providers = append([]string(nil), c.Providers...)
	if c.CUDA.ConvUseMaxWorkspace != nil {
		v := *c.CUDA.ConvUseMaxWorkspace
		c.CUDA.ConvUseMaxWorkspace = &v
	}
	c.CUDA.Extra = maps.Clone(c.CUDA.Extra)
	c.TensorRT.Extra = maps.Clone(c.TensorRT.Extra)
	if c.Placement.CPUOps != nil {
		c.Placement.CPUOps = append([]string{}, c.Placement.CPUOps...)
	}
	return c
}
