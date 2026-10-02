package onnx

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
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
}

// NewRuntime returns a runtime for the configuration, or the reason it cannot
// be built. It loads nothing: Init does.
func NewRuntime(cfg Config) (*Runtime, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Runtime{cfg: cfg.withDefaults(), sessions: map[*Session]struct{}{}}, nil
}

// Config returns the configuration the runtime uses, with defaults filled in.
func (r *Runtime) Config() Config {
	cfg := r.cfg
	cfg.Providers = append([]string(nil), r.cfg.Providers...)
	return cfg
}

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

// Version is the version of the loaded ONNX Runtime library, such as 1.27.1.
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

// probeProvider appends one provider to options nobody uses.
func probeProvider(name string) error {
	if name == ProviderCPU {
		return nil
	}
	options, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	defer options.Destroy()
	return appendProvider(options, name)
}

// appendProvider registers one accelerator on a set of session options.
func appendProvider(options *ort.SessionOptions, name string) error {
	switch name {
	case ProviderCoreML:
		return options.AppendExecutionProviderCoreMLV2(map[string]string{})
	case ProviderCUDA:
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return err
		}
		defer cuda.Destroy()
		return options.AppendExecutionProviderCUDA(cuda)
	default:
		return nil
	}
}

// Open verifies a model file against its manifest and opens a session on it.
//
// The order is the point. A blocked manifest is refused before the file is
// read. The file is read once, and its SHA-256 is compared with the manifest's
// before ONNX Runtime sees a byte of it; the session is then built from those
// same bytes, so what was verified is what runs. The graph's inputs and
// outputs are compared with the manifest's before the session is returned.
//
// A model whose graph keeps its weights in external data files cannot be
// opened this way, because a session built from memory does not read them.
func (r *Runtime) Open(m Manifest, dir string) (*Session, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if m.Status == StatusBlocked {
		return nil, fmt.Errorf("%w: %s %s is blocked by its manifest", ErrModelBlocked, m.ID, m.Version)
	}

	r.mu.RLock()
	defer r.mu.RUnlock()
	if err := r.usable(); err != nil {
		return nil, err
	}

	path := filepath.Join(dir, m.File)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: reading model %s: %w", m.ID, err)
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != m.SHA256 {
		return nil, fmt.Errorf("%w: %s at %s has sha256 %s and its manifest records %s",
			ErrDigestMismatch, m.ID, path, got, m.SHA256)
	}
	if err := checkContract(m, data); err != nil {
		return nil, err
	}

	dynamic, provider, err := r.createSession(m, data)
	if err != nil {
		return nil, err
	}

	s := &Session{
		runtime:  r,
		manifest: cloneManifest(m),
		session:  dynamic,
		info: ModelInfo{
			ID:        m.ID,
			Version:   m.Version,
			SHA256:    m.SHA256,
			Runtime:   "onnxruntime " + r.version,
			Precision: m.Precision,
			Provider:  provider,
			Status:    m.Status,
		},
	}
	r.sessionsMu.Lock()
	r.sessions[s] = struct{}{}
	r.sessionsMu.Unlock()
	return s, nil
}

// createSession builds the session with the available accelerators, and
// falls back to the CPU alone when the library refuses the graph with them.
// It returns the provider the session was built for.
func (r *Runtime) createSession(m Manifest, data []byte) (*ort.DynamicAdvancedSession, string, error) {
	accelerators := []string{}
	for _, status := range r.providers {
		if status.Available && status.Name != ProviderCPU {
			accelerators = append(accelerators, status.Name)
		}
	}

	var firstErr error
	if len(accelerators) > 0 {
		session, err := r.newSession(m, data, accelerators)
		if err == nil {
			return session, accelerators[0], nil
		}
		firstErr = err
	}
	session, err := r.newSession(m, data, nil)
	if err != nil {
		if firstErr != nil {
			err = errors.Join(firstErr, err)
		}
		return nil, "", fmt.Errorf("onnx: creating a session for %s: %w", m.ID, err)
	}
	return session, ProviderCPU, nil
}

// newSession builds one session with the given accelerators in front of the
// CPU.
func (r *Runtime) newSession(m Manifest, data []byte, accelerators []string) (*ort.DynamicAdvancedSession, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()
	if r.cfg.IntraOpThreads > 0 {
		if err := options.SetIntraOpNumThreads(r.cfg.IntraOpThreads); err != nil {
			return nil, err
		}
	}
	for _, name := range accelerators {
		if err := appendProvider(options, name); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	}
	return ort.NewDynamicAdvancedSessionWithONNXData(data, names(m.Inputs), names(m.Outputs), options)
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
	return m
}
