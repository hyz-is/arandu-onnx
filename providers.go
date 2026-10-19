package onnx

import (
	"fmt"
	"maps"
	"path/filepath"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-onnx/calibration"
)

// SessionPlan is what a session of one model on one device is built with:
// the accelerators to try, in order, and the options handed to ONNX Runtime
// for each. It is computed from the configuration and the manifest alone, so
// it can be read, logged and tested on a machine with no GPU.
type SessionPlan struct {
	// Model is the manifest's id.
	Model string
	// Device is the GPU the CUDA and TensorRT providers are given.
	Device int
	// Accelerators is the configured accelerators in order of preference,
	// before Runtime.Init found out which are available.
	Accelerators []string
	// CUDA is the ProviderCUDA options, by ONNX Runtime's names. Empty when
	// ProviderCUDA is not configured.
	CUDA map[string]string
	// TensorRT is the ProviderTensorRT options, by ONNX Runtime's names. Empty
	// when ProviderTensorRT is not configured.
	TensorRT map[string]string
	// INT8 is whether TensorRT is asked to run the model in INT8.
	INT8 bool
	// Calibration is the SHA-256 of the table TensorRT is handed. Empty when
	// there is none.
	Calibration string
	// CalibrationFormat is that table's format.
	CalibrationFormat calibration.Format
	// EngineCacheDir is this model's directory under TensorRTOptions.CacheDir,
	// where its engines and the verified copy of its table are kept. Empty
	// when TensorRT keeps nothing on disk for it.
	EngineCacheDir string
	// CalibrationPath is where the verified copy of the table is written,
	// inside EngineCacheDir. Empty when there is no table.
	CalibrationPath string
}

// The file names a verified table is written under, by format. TensorRT is
// given the name, and finds it in EngineCacheDir.
var calibrationFileNames = map[calibration.Format]string{
	calibration.FormatONNXRuntime: "calibration.flatbuffers",
	calibration.FormatTensorRT:    "calibration.cache",
}

// Plan returns the plan of a session of the model on the device, for the
// configuration with its defaults filled in. It reads no file and loads no
// library.
func (c Config) Plan(m Manifest, device int) (SessionPlan, error) {
	if err := c.Validate(); err != nil {
		return SessionPlan{}, err
	}
	if err := m.Validate(); err != nil {
		return SessionPlan{}, err
	}
	if err := validateDevice("the device", device); err != nil {
		return SessionPlan{}, err
	}
	return c.withDefaults().plan(m, device), nil
}

// plan is Plan for a configuration already validated and defaulted.
func (c Config) plan(m Manifest, device int) SessionPlan {
	p := SessionPlan{Model: m.ID, Device: device}
	for _, name := range c.Providers {
		if name != ProviderCPU {
			p.Accelerators = append(p.Accelerators, name)
		}
	}
	for _, name := range p.Accelerators {
		switch name {
		case ProviderCUDA:
			p.CUDA = c.CUDA.options(device)
		case ProviderTensorRT:
			c.TensorRT.fill(&p, m)
		}
	}
	return p
}

// options is the CUDA provider options for one device.
func (o CUDAOptions) options(device int) map[string]string {
	out := maps.Clone(o.Extra)
	if out == nil {
		out = map[string]string{}
	}
	out["device_id"] = strconv.Itoa(device)
	if o.GPUMemLimit > 0 {
		out["gpu_mem_limit"] = strconv.FormatInt(o.GPUMemLimit, 10)
	}
	if o.ConvAlgoSearch != "" {
		out["cudnn_conv_algo_search"] = o.ConvAlgoSearch
	}
	if o.ConvUseMaxWorkspace != nil {
		out["cudnn_conv_use_max_workspace"] = flag(*o.ConvUseMaxWorkspace)
	}
	if o.ArenaExtendStrategy != "" {
		out["arena_extend_strategy"] = o.ArenaExtendStrategy
	}
	if o.PreferNHWC {
		out["prefer_nhwc"] = "1"
	}
	return out
}

// fill writes the TensorRT options of one model on one device into the plan.
func (o TensorRTOptions) fill(p *SessionPlan, m Manifest) {
	out := maps.Clone(o.Extra)
	if out == nil {
		out = map[string]string{}
	}
	out["device_id"] = strconv.Itoa(p.Device)
	out["trt_fp16_enable"] = flag(o.FP16)

	// INT8 needs something to take the ranges from: the scales of a QDQ graph,
	// or the table of a float one. A float model with neither would be built
	// with no range for any tensor.
	table := m.Calibration != nil && m.Precision != PrecisionINT8
	p.INT8 = o.INT8 && (table || m.Precision == PrecisionINT8)
	out["trt_int8_enable"] = flag(p.INT8)

	if o.CacheDir != "" && (o.EngineCache || p.INT8) {
		name := m.ID + "-" + m.SHA256[:16]
		if p.INT8 && table {
			name += "-" + m.Calibration.SHA256[:16]
		}
		p.EngineCacheDir = filepath.Join(o.CacheDir, name)
		out["trt_engine_cache_path"] = p.EngineCacheDir
	}
	out["trt_engine_cache_enable"] = flag(o.EngineCache)

	if p.INT8 && table {
		p.Calibration = m.Calibration.SHA256
		p.CalibrationFormat = m.Calibration.Format
		file := calibrationFileNames[m.Calibration.Format]
		p.CalibrationPath = filepath.Join(p.EngineCacheDir, file)
		out["trt_int8_calibration_table_name"] = file
		out["trt_int8_use_native_calibration_table"] = flag(m.Calibration.Format == calibration.FormatTensorRT)
	}

	out["trt_timing_cache_enable"] = flag(o.TimingCache)
	if o.TimingCache {
		out["trt_timing_cache_path"] = o.CacheDir
	}
	if o.MaxWorkspaceSize > 0 {
		out["trt_max_workspace_size"] = strconv.FormatInt(o.MaxWorkspaceSize, 10)
	}
	if o.BuilderOptimizationLevel > 0 {
		out["trt_builder_optimization_level"] = strconv.Itoa(o.BuilderOptimizationLevel)
	}
	p.TensorRT = out
}

func flag(v bool) string {
	if v {
		return "1"
	}
	return "0"
}

// probeProviders asks the loaded library to accept each provider on a
// throwaway set of session options.
func probeProviders(preferred []string) []ProviderStatus {
	out := make([]ProviderStatus, 0, len(preferred)+1)
	cpu := false
	for _, name := range preferred {
		if name == ProviderCPU {
			cpu = true
		}
		status := ProviderStatus{Name: name, Available: true}
		if err := probeProvider(name); err != nil {
			status.Available = false
			status.Reason = err.Error()
		}
		out = append(out, status)
	}
	if !cpu {
		out = append(out, ProviderStatus{Name: ProviderCPU, Available: true})
	}
	return out
}

// probeProvider appends one provider, with ONNX Runtime's default options, to
// options nobody uses.
func probeProvider(name string) error {
	if name == ProviderCPU {
		return nil
	}
	options, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	defer options.Destroy()
	return appendProvider(options, name, nil)
}

// appendProvider registers one accelerator on a set of session options, with
// the options the plan gives it. A nil plan is ONNX Runtime's defaults.
func appendProvider(options *ort.SessionOptions, name string, plan *SessionPlan) error {
	switch name {
	case ProviderCoreML:
		return options.AppendExecutionProviderCoreMLV2(map[string]string{})
	case ProviderCUDA:
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return err
		}
		defer cuda.Destroy()
		if plan != nil {
			if err := cuda.Update(plan.CUDA); err != nil {
				return fmt.Errorf("options %v: %w", plan.CUDA, err)
			}
		}
		return options.AppendExecutionProviderCUDA(cuda)
	case ProviderTensorRT:
		trt, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			return err
		}
		defer trt.Destroy()
		if plan != nil {
			if err := trt.Update(plan.TensorRT); err != nil {
				return fmt.Errorf("options %v: %w", plan.TensorRT, err)
			}
		}
		return options.AppendExecutionProviderTensorRT(trt)
	default:
		return nil
	}
}
