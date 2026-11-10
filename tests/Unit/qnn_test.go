package unit_test

import (
	"strings"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
)

func qnnConfig() onnx.Config {
	off := false
	return onnx.Config{
		Providers: []string{onnx.ProviderQNN, onnx.ProviderCPU},
		QNN: onnx.QNNOptions{
			LibraryPath:                          "/usr/lib/libprovider_plugin.so",
			BackendPath:                          "libQnnHtp.so",
			HTPPerformanceMode:                   "burst",
			HTPGraphFinalizationOptimizationMode: 3,
			VTCMMB:                               8,
			HTPArch:                              68,
			SocModel:                             57,
			OffloadGraphIOQuantization:           &off,
			EnableHTPSharedMemoryAllocator:       true,
			ContextCache:                         true,
			CacheDir:                             "/var/cache/app/qnn",
			EmbedContext:                         true,
			DisableCPUFallback:                   true,
			Extra:                                map[string]string{"rpc_control_latency": "100"},
		},
	}
}

func TestTheQNNPlanCarriesEveryOptionAndEntry(t *testing.T) {
	m := floatManifest(t)
	plan, err := qnnConfig().Plan(m, 0)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	wantOptions := map[string]string{
		"backend_path":                             "libQnnHtp.so",
		"htp_performance_mode":                     "burst",
		"htp_graph_finalization_optimization_mode": "3",
		"vtcm_mb":                            "8",
		"htp_arch":                           "68",
		"soc_model":                          "57",
		"offload_graph_io_quantization":      "0",
		"enable_htp_shared_memory_allocator": "1",
		"rpc_control_latency":                "100",
	}
	if len(plan.QNN) != len(wantOptions) {
		t.Errorf("QNN options = %v, want %v", plan.QNN, wantOptions)
	}
	for k, v := range wantOptions {
		if plan.QNN[k] != v {
			t.Errorf("QNN[%s] = %q, want %q", k, plan.QNN[k], v)
		}
	}
	wantFile := "/var/cache/app/qnn/" + m.ID + "-" + m.SHA256[:16] + "-qnn_ctx.onnx"
	if plan.QNNContextFile != wantFile {
		t.Errorf("QNNContextFile = %q, want %q", plan.QNNContextFile, wantFile)
	}
	wantEntries := map[string]string{
		"ep.context_enable":               "1",
		"ep.context_file_path":            wantFile,
		"ep.context_embed_mode":           "1",
		"session.disable_cpu_ep_fallback": "1",
	}
	for k, v := range wantEntries {
		if plan.QNNConfig[k] != v {
			t.Errorf("QNNConfig[%s] = %q, want %q", k, plan.QNNConfig[k], v)
		}
	}
	if len(plan.Accelerators) != 1 || plan.Accelerators[0] != onnx.ProviderQNN {
		t.Errorf("Accelerators = %v, want [qnn]", plan.Accelerators)
	}
	if plan.CUDA != nil || plan.TensorRT != nil {
		t.Error("a QNN plan carries CUDA or TensorRT options")
	}
}

func TestQNNNamesItsBackendByTypeOrByPath(t *testing.T) {
	cfg := onnx.Config{
		Providers: []string{onnx.ProviderQNN},
		QNN:       onnx.QNNOptions{LibraryPath: "libplugin.so", BackendType: "htp", HTPArch: 68, HTPPerformanceMode: "burst"},
	}
	plan, err := cfg.Plan(floatManifest(t), 0)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if plan.QNN["backend_type"] != "htp" || plan.QNN["backend_path"] != "" || plan.QNN["vtcm_mb"] != "" {
		t.Errorf("QNN options = %v, want backend_type htp and no backend_path or vtcm_mb", plan.QNN)
	}
	cfg.QNN.BackendPath = "libQnnHtp.so"
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "both set") {
		t.Errorf("both backends = %v, want a refusal", err)
	}
}

func TestAMinimalQNNPlanSetsOnlyTheBackend(t *testing.T) {
	cfg := onnx.Config{
		Providers: []string{onnx.ProviderQNN},
		QNN:       onnx.QNNOptions{LibraryPath: "libplugin.so", BackendPath: "libQnnHtp.so"},
	}
	plan, err := cfg.Plan(floatManifest(t), 0)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if len(plan.QNN) != 1 || plan.QNN["backend_path"] != "libQnnHtp.so" {
		t.Errorf("QNN options = %v, want only backend_path", plan.QNN)
	}
	if len(plan.QNNConfig) != 0 || plan.QNNContextFile != "" {
		t.Errorf("a minimal plan sets entries %v and a context file %q", plan.QNNConfig, plan.QNNContextFile)
	}
}

func TestQNNIsNeverInThePlanOfAConfigThatDoesNotNameIt(t *testing.T) {
	plan, err := gpuConfig().Plan(floatManifest(t), 1)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if plan.QNN != nil || plan.QNNConfig != nil || plan.QNNLibrary != "" {
		t.Errorf("a GPU plan carries QNN settings: %+v", plan)
	}
}

func TestQNNConfigurationMistakesAreRefusedWhereTheyAreWired(t *testing.T) {
	cases := map[string]struct {
		change func(*onnx.QNNOptions)
		want   string
	}{
		"no library":   {func(o *onnx.QNNOptions) { o.LibraryPath = "" }, "LibraryPath"},
		"no backend":   {func(o *onnx.QNNOptions) { o.BackendPath = "" }, "BackendPath"},
		"bad mode":     {func(o *onnx.QNNOptions) { o.HTPGraphFinalizationOptimizationMode = 4 }, "HTPGraphFinalizationOptimizationMode"},
		"negative":     {func(o *onnx.QNNOptions) { o.VTCMMB = -1 }, "negative"},
		"relative dir": {func(o *onnx.QNNOptions) { o.CacheDir = "cache" }, "absolute"},
		"cache no dir": {func(o *onnx.QNNOptions) { o.CacheDir = "" }, "CacheDir"},
		"embed alone":  {func(o *onnx.QNNOptions) { o.ContextCache = false }, "ContextCache"},
		"managed key":  {func(o *onnx.QNNOptions) { o.Extra = map[string]string{"vtcm_mb": "4"} }, "vtcm_mb"},
		"saver path":   {func(o *onnx.QNNOptions) { o.Extra = map[string]string{"qnn_saver_path": "x.so"} }, "qnn_saver_path"},
		"profile file": {func(o *onnx.QNNOptions) { o.Extra = map[string]string{"profiling_file_path": "/tmp/x"} }, "profiling_file_path"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := qnnConfig()
			tc.change(&cfg.QNN)
			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Validate = %v, want an error naming %q", err, tc.want)
			}
		})
	}
	if err := qnnConfig().Validate(); err != nil {
		t.Errorf("the complete configuration is refused: %v", err)
	}
	// Not naming the provider, the QNN settings are not read.
	if err := (onnx.Config{QNN: onnx.QNNOptions{BackendPath: "x"}}).Validate(); err != nil {
		t.Errorf("QNN settings without the provider are refused: %v", err)
	}
}

func TestRequiringAcceleratedAcceptsQNNAsTheAccelerator(t *testing.T) {
	cfg := qnnConfig()
	cfg.Placement.RequireAccelerated = true
	if err := cfg.Validate(); err != nil {
		t.Errorf("RequireAccelerated with qnn = %v", err)
	}
}

func TestQNNPlacementIsReadFromTheLog(t *testing.T) {
	log := `[V:onnxruntime:, session_state.cc:1] Node placements
[V:onnxruntime:, session_state.cc:1]  Node(s) placed on [QNNExecutionProvider]. Number of nodes: 3
[V:onnxruntime:, session_state.cc:1]   QNN_Compiled (QNNExecutionProvider_QNN_1_0)
[V:onnxruntime:, session_state.cc:1]  Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1
[V:onnxruntime:, session_state.cc:1]   Shape (/Shape)
`
	p := onnx.ParsePlacement(log)
	if p.Nodes[onnx.LogProviderQNN] != 3 || p.Nodes[onnx.LogProviderCPU] != 1 {
		t.Fatalf("Nodes = %v", p.Nodes)
	}
	if fb := p.Fallback(onnx.DefaultCPUOps()); len(fb) != 0 {
		t.Errorf("shape arithmetic on the CPU counted as a fallback: %v", fb)
	}
}
