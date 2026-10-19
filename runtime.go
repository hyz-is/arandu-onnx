package onnx

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// The errors a Runtime reports. Each is wrapped with the detail of the case,
// so callers test for them with errors.Is.
var (
	// ErrLibraryUnavailable is the ONNX Runtime shared library missing, not
	// loadable, or too old for the API this package requests.
	ErrLibraryUnavailable = errors.New("onnx: ONNX Runtime library unavailable")
	// ErrNotInitialized is a runtime used before Init.
	ErrNotInitialized = errors.New("onnx: runtime not initialized")
	// ErrClosed is a runtime or a session used after Close.
	ErrClosed = errors.New("onnx: closed")
	// ErrModelBlocked is a manifest whose status is blocked.
	ErrModelBlocked = errors.New("onnx: model blocked")
	// ErrDigestMismatch is a model file whose SHA-256 is not the manifest's.
	ErrDigestMismatch = errors.New("onnx: model digest mismatch")
	// ErrContractMismatch is a model graph whose inputs or outputs are not the
	// ones its manifest declares.
	ErrContractMismatch = errors.New("onnx: model contract mismatch")
)

// environment is the one ONNX Runtime environment of the process.
//
// The binding keeps the loaded library and the environment in package globals,
// so there is one per process whatever the number of Runtime values. They
// share it: the first Init loads the library, the last Close unloads it, and a
// second library path in the same process is refused rather than ignored.
var environment struct {
	sync.Mutex
	owners  int
	library string
	version string
}

// acquireEnvironment loads the library and creates the environment, or joins
// the one that exists. It returns the library's version.
func acquireEnvironment(library string) (string, error) {
	environment.Lock()
	defer environment.Unlock()

	if environment.owners > 0 {
		if environment.library != library {
			return "", fmt.Errorf("onnx: ONNX Runtime is already loaded from %s in this process, and a process loads one library: this runtime asks for %s",
				environment.library, library)
		}
		environment.owners++
		return environment.version, nil
	}

	// A path is checked before it is handed to the loader, so a missing file
	// is reported as one rather than as whatever the loader says about it. A
	// bare file name is left to the loader's own search path.
	if strings.ContainsAny(library, `/\`) {
		if _, err := os.Stat(library); err != nil {
			return "", fmt.Errorf("%w: %v", ErrLibraryUnavailable, err)
		}
	}
	ort.SetSharedLibraryPath(library)
	if err := ort.InitializeEnvironment(); err != nil {
		return "", fmt.Errorf("%w: %s: %v", ErrLibraryUnavailable, library, err)
	}
	environment.owners = 1
	environment.library = library
	environment.version = ort.GetVersion()
	return environment.version, nil
}

// releaseEnvironment gives one share of the environment back, and destroys it
// with the last one.
func releaseEnvironment() error {
	environment.Lock()
	defer environment.Unlock()

	if environment.owners == 0 {
		return nil
	}
	environment.owners--
	if environment.owners > 0 {
		return nil
	}
	environment.library = ""
	environment.version = ""
	return ort.DestroyEnvironment()
}

// ProviderStatus is what Init found out about one execution provider.
type ProviderStatus struct {
	// Name is the provider, as Config names it.
	Name string
	// Available is whether the loaded library accepted it.
	Available bool
	// Reason is why it was not available. Empty when it was.
	Reason string
}

// runtimeState is where a Runtime is in its life.
type runtimeState int

const (
	stateNew runtimeState = iota
	stateReady
	stateClosed
)

// Runtime is a configured ONNX Runtime and the sessions opened through it.
//
// Its methods are safe for concurrent use. Init and Close are idempotent; a
// Runtime that was closed stays closed. Close waits for every Open in flight,
// then closes every session, each of which waits for its own runs.
type Runtime struct {
	cfg Config

	// mu orders the life of the runtime: Init and Close hold it for writing,
	// Open holds it for reading for as long as it creates a session, so the
	// environment cannot be destroyed under a session being built.
	mu        sync.RWMutex
	state     runtimeState
	version   string
	providers []ProviderStatus

	sessionsMu sync.Mutex
	sessions   map[*Session]struct{}

	// placements is the placement captured for each model on each device,
	// by placementKey, so only the first session of each is built verbose.
	placementsMu sync.Mutex
	placements   map[string]Placement
}

// NewRuntime returns a runtime for the configuration, or the reason it cannot
// be built. It loads nothing: Init does.
func NewRuntime(cfg Config) (*Runtime, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Runtime{
		cfg:        cfg.withDefaults(),
		sessions:   map[*Session]struct{}{},
		placements: map[string]Placement{},
	}, nil
}

// Config returns the configuration the runtime uses, with defaults filled in.
func (r *Runtime) Config() Config { return r.cfg.clone() }

// Init loads the shared library, creates the environment and probes the
// configured providers. A second call on a ready runtime does nothing; a call
// on a closed one returns ErrClosed. A failed Init leaves the runtime as it
// was, so it may be retried.
func (r *Runtime) Init() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch r.state {
	case stateReady:
		return nil
	case stateClosed:
		return ErrClosed
	}
	version, err := acquireEnvironment(r.cfg.SharedLibraryPath)
	if err != nil {
		return err
	}
	r.version = version
	r.providers = probeProviders(r.cfg.Providers)
	r.state = stateReady
	return nil
}

// Close closes every session opened through the runtime and gives the
// environment back. It is idempotent, and a second call returns once the first
// has finished.
func (r *Runtime) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state == stateClosed {
		return nil
	}
	wasReady := r.state == stateReady
	r.state = stateClosed

	r.sessionsMu.Lock()
	open := make([]*Session, 0, len(r.sessions))
	for s := range r.sessions {
		open = append(open, s)
	}
	r.sessionsMu.Unlock()

	var errs []error
	for _, s := range open {
		errs = append(errs, s.Close())
	}
	if wasReady {
		errs = append(errs, releaseEnvironment())
	}
	return errors.Join(errs...)
}

// Health reports whether the runtime can open sessions: nil once Init has
// succeeded, ErrNotInitialized before, ErrClosed after Close.
func (r *Runtime) Health() error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.usable()
}

// usable is Health for a caller already holding mu.
func (r *Runtime) usable() error {
	switch r.state {
	case stateReady:
		return nil
	case stateClosed:
		return ErrClosed
	default:
		return ErrNotInitialized
	}
}

// Version is the version of the loaded ONNX Runtime library, such as 1.30.0.
// It is empty before Init.
func (r *Runtime) Version() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.version
}

// Providers reports, in order of preference, what Init found out about each
// configured provider, with the CPU last when the configuration did not name
// it. It is empty before Init.
func (r *Runtime) Providers() []ProviderStatus {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]ProviderStatus(nil), r.providers...)
}

// checkContract compares the graph's inputs and outputs with the manifest's.
//
// Every input of the graph must be declared, because a session cannot run
// with one missing. Every declared output must exist; the graph may have more,
// and those are not fetched. A dimension both sides fix must agree.
func checkContract(m Manifest, data []byte) error {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(data)
	if err != nil {
		return fmt.Errorf("%w: %s: reading the graph: %v", ErrContractMismatch, m.ID, err)
	}
	if len(inputs) != len(m.Inputs) {
		return fmt.Errorf("%w: %s: the graph has %d inputs and the manifest declares %d",
			ErrContractMismatch, m.ID, len(inputs), len(m.Inputs))
	}
	if err := matchSpecs(m.ID, "input", m.Inputs, inputs); err != nil {
		return err
	}
	return matchSpecs(m.ID, "output", m.Outputs, outputs)
}

// matchSpecs finds each declared tensor in the graph and compares it.
func matchSpecs(id, side string, declared []TensorSpec, graph []ort.InputOutputInfo) error {
	byName := make(map[string]ort.InputOutputInfo, len(graph))
	for _, info := range graph {
		byName[info.Name] = info
	}
	for _, spec := range declared {
		info, ok := byName[spec.Name]
		if !ok {
			return fmt.Errorf("%w: %s: the graph has no %s %q", ErrContractMismatch, id, side, spec.Name)
		}
		if info.OrtValueType != ort.ONNXTypeTensor {
			return fmt.Errorf("%w: %s: %s %q is a %s, not a tensor", ErrContractMismatch, id, side, spec.Name, info.OrtValueType)
		}
		if !sameElementType(spec.ElementType, info.DataType) {
			return fmt.Errorf("%w: %s: %s %q is %s in the graph and %s in the manifest",
				ErrContractMismatch, id, side, spec.Name, info.DataType, spec.ElementType)
		}
		if len(info.Dimensions) != len(spec.Shape) {
			return fmt.Errorf("%w: %s: %s %q has shape %v in the graph and %v in the manifest",
				ErrContractMismatch, id, side, spec.Name, []int64(info.Dimensions), spec.Shape)
		}
		for i, dim := range spec.Shape {
			if graphDim := info.Dimensions[i]; dim != -1 && graphDim > 0 && graphDim != dim {
				return fmt.Errorf("%w: %s: %s %q has shape %v in the graph and %v in the manifest",
					ErrContractMismatch, id, side, spec.Name, []int64(info.Dimensions), spec.Shape)
			}
		}
	}
	return nil
}

// sameElementType maps this package's element types onto the binding's.
func sameElementType(kind ElementType, graph ort.TensorElementDataType) bool {
	switch kind {
	case Float32:
		return graph == ort.TensorElementDataTypeFloat
	case Int64:
		return graph == ort.TensorElementDataTypeInt64
	case Uint8:
		return graph == ort.TensorElementDataTypeUint8
	default:
		return false
	}
}

// cloneManifest copies the slices a session keeps, so a caller changing its
// manifest afterwards does not change the session's contract.
func cloneManifest(m Manifest) Manifest {
	clone := func(specs []TensorSpec) []TensorSpec {
		out := make([]TensorSpec, len(specs))
		for i, spec := range specs {
			spec.Shape = append([]int64(nil), spec.Shape...)
			out[i] = spec
		}
		return out
	}
	m.Inputs = clone(m.Inputs)
	m.Outputs = clone(m.Outputs)
	if m.Calibration != nil {
		c := *m.Calibration
		m.Calibration = &c
	}
	return m
}
