package onnx

import (
	"fmt"
	"runtime"
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
)

// MaxIntraOpThreads is the ceiling Config.IntraOpThreads is refused above. A
// number past any machine's core count is a typo, not a tuning decision.
const MaxIntraOpThreads = 1024

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
	// ProviderCPU, ProviderCoreML, ProviderCUDA. Empty means ProviderCPU alone.
	//
	// A provider the loaded library or the machine cannot offer is skipped and
	// reported by Runtime.Providers, not refused: the CPU is always the last
	// resort, so naming an accelerator is a preference and never a reason for
	// the application not to start.
	Providers []string
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
	for _, provider := range c.Providers {
		switch provider {
		case ProviderCPU, ProviderCoreML, ProviderCUDA:
		default:
			return fmt.Errorf("onnx: Config.Providers names %q, which is not one of %q, %q or %q",
				provider, ProviderCPU, ProviderCoreML, ProviderCUDA)
		}
		if seen[provider] {
			return fmt.Errorf("onnx: Config.Providers names %q twice", provider)
		}
		seen[provider] = true
	}
	return nil
}

// withDefaults returns the configuration with the optional fields filled in.
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
	c.Providers = append([]string(nil), c.Providers...)
	return c
}
