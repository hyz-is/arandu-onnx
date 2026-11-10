package onnx

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// LogProviderQNN is how ONNX Runtime names ProviderQNN in its logs and in the
// plugin's device list.
const LogProviderQNN = "QNNExecutionProvider"

// DefaultQNNRegistrationName is the name the plugin library is registered
// under when QNNOptions.RegistrationName is empty. It is the provider's own
// name: the Qualcomm plugin lists its NPU device only when it is registered
// under it, and under any other name the device list has no QNN entry
// (measured on a QCS6490 with onnxruntime-qnn 2.7.0 and ONNX Runtime 1.27.1).
const DefaultQNNRegistrationName = LogProviderQNN

// QNNOptions is the ProviderQNN configuration: the Qualcomm AI Engine Direct
// execution provider, loaded as a plugin library and run on the Hexagon NPU.
//
// Nothing here is discovered. The plugin library, the backend library and the
// cache directory are named by the application, and a provider that cannot be
// loaded is reported by Runtime.Providers. To refuse a session that did not
// run on the NPU, set Config.Placement.RequireAccelerated, and
// DisableCPUFallback to make ONNX Runtime itself refuse a graph the provider
// cannot take whole.
//
// A zero field leaves the plugin's default. The fields name ONNX Runtime's
// own provider options and session configuration entries.
type QNNOptions struct {
	// LibraryPath is the plugin execution provider library, such as
	// libonnxruntime_providers_qnn.so. Required. It is registered once per
	// process.
	LibraryPath string

	// RegistrationName is the name the library is registered under. Empty
	// means DefaultQNNRegistrationName. A process registers one library per
	// name; naming the same one with another path is refused.
	RegistrationName string

	// BackendPath is the QNN backend library, such as libQnnHtp.so
	// (provider option backend_path). One of BackendPath and BackendType is
	// required.
	BackendPath string

	// BackendType is the QNN backend by name, such as "htp" (provider option
	// backend_type), for a plugin that finds the library itself. It is
	// refused beside BackendPath.
	BackendType string

	// HTPPerformanceMode is htp_performance_mode: "burst", "balanced",
	// "default", "high_performance", "high_power_saver", "low_balanced",
	// "low_power_saver", "power_saver", "extreme_power_saver" or
	// "sustained_high_performance".
	HTPPerformanceMode string

	// HTPGraphFinalizationOptimizationMode is
	// htp_graph_finalization_optimization_mode, 0 to 3. Zero leaves the
	// default; higher values build slower for a faster graph.
	HTPGraphFinalizationOptimizationMode int

	// VTCMMB is vtcm_mb, the tightly coupled memory the graph may use, in
	// megabytes. Zero leaves the default, which is what to leave on a
	// plugin build that rejects the option: it is set only when positive.
	VTCMMB int

	// HTPArch is htp_arch, the Hexagon architecture, such as 68. Zero leaves
	// the default.
	HTPArch int

	// SocModel is soc_model. Zero leaves the default.
	SocModel int

	// OffloadGraphIOQuantization is offload_graph_io_quantization: the
	// quantization of the graph's inputs and outputs is done by the
	// provider. Nil leaves the default.
	OffloadGraphIOQuantization *bool

	// EnableHTPSharedMemoryAllocator is enable_htp_shared_memory_allocator.
	EnableHTPSharedMemoryAllocator bool

	// ContextCache keeps the compiled context binary in CacheDir
	// (session option ep.context_enable) and loads it on the next start
	// instead of compiling the graph again. Compiling takes seconds to
	// minutes. The cached model is specific to the hardware generation it
	// was compiled for.
	ContextCache bool

	// CacheDir is an absolute, writable directory, private to the
	// application, that holds the compiled context of each model under a name
	// made of the manifest's id and the digest of its file. Required by
	// ContextCache.
	CacheDir string

	// EmbedContext is ep.context_embed_mode: the compiled binary is embedded
	// in the cached model instead of written beside it.
	EmbedContext bool

	// DisableCPUFallback is session.disable_cpu_ep_fallback: building the
	// session fails when any node cannot be placed on the provider, instead
	// of running it on the CPU.
	DisableCPUFallback bool

	// DeviceIndex picks among the devices the plugin lists, in the order
	// ONNX Runtime returns them. Zero is the first.
	DeviceIndex int

	// Extra is any other provider option, by ONNX Runtime's name. A key one
	// of the fields above sets is refused, and so are the options that make
	// the provider load a library or write a file elsewhere: qnn_saver_path
	// and profiling_file_path.
	Extra map[string]string
}

// managedQNNOptions is the provider option names a QNNOptions field sets.
var managedQNNOptions = map[string]bool{
	"backend_path":         true,
	"backend_type":         true,
	"htp_performance_mode": true,
	"htp_graph_finalization_optimization_mode": true,
	"vtcm_mb":                            true,
	"htp_arch":                           true,
	"soc_model":                          true,
	"offload_graph_io_quantization":      true,
	"enable_htp_shared_memory_allocator": true,
}

// refusedQNNOptions is the provider option names that would make the provider
// load a library or write outside CacheDir.
var refusedQNNOptions = map[string]bool{
	"qnn_saver_path":      true,
	"profiling_file_path": true,
}

func (o QNNOptions) validate() error {
	if strings.TrimSpace(o.LibraryPath) == "" {
		return fmt.Errorf("onnx: Config.QNN.LibraryPath is required when Config.Providers names %q: it is the plugin execution provider library", ProviderQNN)
	}
	if strings.TrimSpace(o.BackendPath) == "" && strings.TrimSpace(o.BackendType) == "" {
		return fmt.Errorf("onnx: Config.QNN.BackendPath or BackendType is required when Config.Providers names %q: it names the QNN backend", ProviderQNN)
	}
	if o.BackendPath != "" && o.BackendType != "" {
		return fmt.Errorf("onnx: Config.QNN.BackendPath and BackendType are both set, and a backend is named once")
	}
	if o.HTPGraphFinalizationOptimizationMode < 0 || o.HTPGraphFinalizationOptimizationMode > 3 {
		return fmt.Errorf("onnx: Config.QNN.HTPGraphFinalizationOptimizationMode is %d, and has to be 0 (the default) or 1 to 3", o.HTPGraphFinalizationOptimizationMode)
	}
	if o.VTCMMB < 0 || o.HTPArch < 0 || o.SocModel < 0 || o.DeviceIndex < 0 {
		return fmt.Errorf("onnx: Config.QNN.VTCMMB, HTPArch, SocModel and DeviceIndex cannot be negative")
	}
	if o.CacheDir != "" && (!filepath.IsAbs(o.CacheDir) || filepath.Clean(o.CacheDir) != o.CacheDir) {
		return fmt.Errorf("onnx: Config.QNN.CacheDir is %q, and has to be a clean absolute path: a relative one would move with the working directory", o.CacheDir)
	}
	if (o.ContextCache || o.EmbedContext) && o.CacheDir == "" {
		return fmt.Errorf("onnx: Config.QNN.ContextCache and EmbedContext need Config.QNN.CacheDir, where the compiled context is kept")
	}
	if o.EmbedContext && !o.ContextCache {
		return fmt.Errorf("onnx: Config.QNN.EmbedContext needs Config.QNN.ContextCache")
	}
	for key := range o.Extra {
		switch {
		case strings.TrimSpace(key) == "":
			return fmt.Errorf("onnx: Config.QNN.Extra has an empty key")
		case managedQNNOptions[key] || key == "device_id":
			return fmt.Errorf("onnx: Config.QNN.Extra sets %q, which a QNNOptions field sets", key)
		case refusedQNNOptions[key]:
			return fmt.Errorf("onnx: Config.QNN.Extra sets %q, which makes the provider load a library or write outside CacheDir, and is refused", key)
		}
	}
	return nil
}

func (o QNNOptions) clone() QNNOptions {
	if o.OffloadGraphIOQuantization != nil {
		v := *o.OffloadGraphIOQuantization
		o.OffloadGraphIOQuantization = &v
	}
	o.Extra = maps.Clone(o.Extra)
	return o
}

func (o QNNOptions) registrationName() string {
	if o.RegistrationName == "" {
		return DefaultQNNRegistrationName
	}
	return o.RegistrationName
}

// fill writes the QNN options of one model into the plan.
func (o QNNOptions) fill(p *SessionPlan, m Manifest) {
	out := maps.Clone(o.Extra)
	if out == nil {
		out = map[string]string{}
	}
	if o.BackendPath != "" {
		out["backend_path"] = o.BackendPath
	}
	if o.BackendType != "" {
		out["backend_type"] = o.BackendType
	}
	if o.HTPPerformanceMode != "" {
		out["htp_performance_mode"] = o.HTPPerformanceMode
	}
	if o.HTPGraphFinalizationOptimizationMode > 0 {
		out["htp_graph_finalization_optimization_mode"] = strconv.Itoa(o.HTPGraphFinalizationOptimizationMode)
	}
	if o.VTCMMB > 0 {
		out["vtcm_mb"] = strconv.Itoa(o.VTCMMB)
	}
	if o.HTPArch > 0 {
		out["htp_arch"] = strconv.Itoa(o.HTPArch)
	}
	if o.SocModel > 0 {
		out["soc_model"] = strconv.Itoa(o.SocModel)
	}
	if o.OffloadGraphIOQuantization != nil {
		out["offload_graph_io_quantization"] = flag(*o.OffloadGraphIOQuantization)
	}
	if o.EnableHTPSharedMemoryAllocator {
		out["enable_htp_shared_memory_allocator"] = "1"
	}
	p.QNN = out

	entries := map[string]string{}
	if o.DisableCPUFallback {
		entries["session.disable_cpu_ep_fallback"] = "1"
	}
	if o.ContextCache && o.CacheDir != "" {
		p.QNNContextFile = filepath.Join(o.CacheDir, m.ID+"-"+m.SHA256[:16]+"-qnn_ctx.onnx")
		entries["ep.context_enable"] = "1"
		entries["ep.context_file_path"] = p.QNNContextFile
		entries["ep.context_embed_mode"] = flag(o.EmbedContext)
	}
	p.QNNConfig = entries
	p.QNNLibrary = o.LibraryPath
	p.QNNRegistration = o.registrationName()
	p.QNNDeviceIndex = o.DeviceIndex
}

// qnnDevice registers the plugin library, once per process, and returns the
// device at the index among the ones the plugin lists.
func qnnDevice(name, library string, index int) (ort.EpDevice, error) {
	if err := registerPlugin(name, library); err != nil {
		return ort.EpDevice{}, err
	}
	devices, err := ort.GetEpDevices()
	if err != nil {
		return ort.EpDevice{}, fmt.Errorf("listing the execution provider devices: %w", err)
	}
	var found []ort.EpDevice
	for _, d := range devices {
		if d.EpName() == LogProviderQNN {
			found = append(found, d)
		}
	}
	if len(found) == 0 {
		return ort.EpDevice{}, fmt.Errorf("the plugin %s lists no %s device: the NPU, its driver or the backend library is not available",
			library, LogProviderQNN)
	}
	if index >= len(found) {
		return ort.EpDevice{}, fmt.Errorf("Config.QNN.DeviceIndex is %d and the plugin lists %d %s devices",
			index, len(found), LogProviderQNN)
	}
	return found[index], nil
}

// registerPlugin registers a plugin library with the environment, once per
// process and name. The registration lives and dies with the environment.
func registerPlugin(name, path string) error {
	environment.Lock()
	defer environment.Unlock()
	if environment.owners == 0 {
		return ErrNotInitialized
	}
	if registered, ok := environment.plugins[name]; ok {
		if registered != path {
			return fmt.Errorf("onnx: the plugin %q is already registered from %s, and this runtime asks for %s", name, registered, path)
		}
		return nil
	}
	if strings.ContainsAny(path, `/\`) {
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("%w: %v", ErrLibraryUnavailable, err)
		}
	}
	if err := ort.RegisterExecutionProviderLibrary(name, path); err != nil {
		return fmt.Errorf("registering the plugin %s: %w", path, err)
	}
	if environment.plugins == nil {
		environment.plugins = map[string]string{}
	}
	environment.plugins[name] = path
	return nil
}

// probeQNN reports why ProviderQNN is not available, or nil.
func probeQNN(o QNNOptions) error {
	_, err := qnnDevice(o.registrationName(), o.LibraryPath, o.DeviceIndex)
	return err
}

// appendQNN registers the plugin execution provider on a set of session
// options, with the session configuration entries the plan asks for. The
// context-writing entries are left out when the session is built from a
// context compiled earlier.
func appendQNN(options *ort.SessionOptions, plan *SessionPlan) error {
	device, err := qnnDevice(plan.QNNRegistration, plan.QNNLibrary, plan.QNNDeviceIndex)
	if err != nil {
		return err
	}
	cached := cachedQNNContext(*plan)
	for key, value := range plan.QNNConfig {
		if cached && strings.HasPrefix(key, "ep.context_") {
			continue
		}
		if err := options.AddSessionConfigEntry(key, value); err != nil {
			return fmt.Errorf("session configuration %s=%s: %w", key, value, err)
		}
	}
	if err := options.AppendExecutionProviderV2([]ort.EpDevice{device}, plan.QNN); err != nil {
		return fmt.Errorf("options %v: %w", plan.QNN, err)
	}
	return nil
}

// prepareQNN creates the directory the compiled context is kept in.
func prepareQNN(plan SessionPlan) error {
	if plan.QNNContextFile == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(plan.QNNContextFile), 0o700); err != nil {
		return fmt.Errorf("onnx: the QNN cache of %s: %w", plan.Model, err)
	}
	return nil
}

// cachedQNNContext reports whether a compiled context from an earlier start
// exists for the plan.
func cachedQNNContext(plan SessionPlan) bool {
	if plan.QNNContextFile == "" {
		return false
	}
	info, err := os.Stat(plan.QNNContextFile)
	return err == nil && info.Mode().IsRegular() && info.Size() > 0
}

// versionQNNContext names the compiled context of a model after everything
// that makes a compiled context stale: the model's digest (already in the
// name), the ONNX Runtime version that compiled it, and the size and
// modification time of the plugin library. Another model, another runtime or
// another plugin build finds no cache and compiles again.
func versionQNNContext(plan *SessionPlan, runtimeVersion string) {
	if plan.QNNContextFile == "" {
		return
	}
	stamp := "unknown"
	if info, err := os.Stat(plan.QNNLibrary); err == nil {
		stamp = strconv.FormatInt(info.Size(), 16) + "-" + strconv.FormatInt(info.ModTime().UnixNano(), 16)
	}
	dir, file := filepath.Split(plan.QNNContextFile)
	base := strings.TrimSuffix(file, "-qnn_ctx.onnx")
	plan.QNNContextFile = filepath.Join(dir, base+"-ort"+runtimeVersion+"-plugin"+stamp+"-qnn_ctx.onnx")
	plan.QNNConfig = maps.Clone(plan.QNNConfig)
	if _, ok := plan.QNNConfig["ep.context_file_path"]; ok {
		plan.QNNConfig["ep.context_file_path"] = plan.QNNContextFile
	}
}
