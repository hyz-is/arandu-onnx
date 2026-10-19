package onnx

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/hyz-is/arandu-onnx/calibration"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/internal/onnxgraph"
	"github.com/hyz-is/arandu-onnx/quantize"
)

// acceleratedBuilds serialises the building of sessions on an accelerator,
// across every Runtime of the process.
//
// Building GPU sessions concurrently on one card has been reported to crash
// ONNX Runtime (onnxruntime issue 26610, CUDA and TensorRT), and a TensorRT
// build is minutes of the card's memory and compute anyway: two at once only
// race for both. Session builds happen at boot, so serialising them costs
// nothing a request waits for.
var acceleratedBuilds sync.Mutex

// Open verifies a model file against its manifest and opens a session on it,
// on Config.Device when the session runs on a GPU.
//
// The order is the point. A blocked manifest is refused before the file is
// read. The file is read once, and its SHA-256 is compared with the manifest's
// before ONNX Runtime sees a byte of it; the session is then built from those
// same bytes, so what was verified is what runs. The graph's inputs and
// outputs are compared with the manifest's before the session is returned. A
// calibration table the manifest carries is verified the same way, and parsed,
// and only the verified bytes are written where TensorRT reads them. A model
// whose manifest declares quantization qdq is checked to be in the QDQ form
// TensorRT runs (quantize.CheckTensorRT).
//
// A model whose graph keeps its weights in external data files cannot be
// opened this way, because a session built from memory does not read them.
func (r *Runtime) Open(m Manifest, dir string) (*Session, error) {
	return r.OpenOn(m, dir, r.cfg.Device)
}

// OpenOn is Open on a given GPU, for the ProviderCUDA and ProviderTensorRT
// sessions. An application with two cards opens each model once per card and
// spreads the work between the sessions.
func (r *Runtime) OpenOn(m Manifest, dir string, device int) (*Session, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := validateDevice("the device", device); err != nil {
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

	data, err := readVerified(m, dir)
	if err != nil {
		return nil, err
	}
	if err := checkContract(m, data); err != nil {
		return nil, err
	}
	if err := checkQuantization(m, data); err != nil {
		return nil, err
	}
	table, err := readCalibration(m, dir, data)
	if err != nil {
		return nil, err
	}

	plan := r.cfg.plan(m, device)
	built, err := r.createSession(m, data, table, plan)
	if err != nil {
		return nil, err
	}

	info := ModelInfo{
		ID:        m.ID,
		Version:   m.Version,
		SHA256:    m.SHA256,
		Runtime:   "onnxruntime " + r.version,
		Precision: m.Precision,
		Provider:  built.provider,
		Status:    m.Status,
	}
	if built.provider == ProviderCUDA || built.provider == ProviderTensorRT {
		info.Device = device
	}
	if built.provider == ProviderTensorRT {
		info.Calibration = plan.Calibration
	}
	s := &Session{
		runtime:   r,
		manifest:  cloneManifest(m),
		session:   built.session,
		info:      info,
		placement: built.placement,
	}
	r.sessionsMu.Lock()
	r.sessions[s] = struct{}{}
	r.sessionsMu.Unlock()
	return s, nil
}

// readVerified reads a model file and checks its digest against the manifest.
func readVerified(m Manifest, dir string) ([]byte, error) {
	path := filepath.Join(dir, m.File)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: reading model %s: %w", m.ID, err)
	}
	if got := digest(data); got != m.SHA256 {
		return nil, fmt.Errorf("%w: %s at %s has sha256 %s and its manifest records %s",
			ErrDigestMismatch, m.ID, path, got, m.SHA256)
	}
	return data, nil
}

// checkQuantization holds a model whose manifest declares QuantizationQDQ to
// the form TensorRT runs, so that a QOperator, dynamic or asymmetric export
// pinned as qdq is refused at Open rather than found on a GPU node.
func checkQuantization(m Manifest, data []byte) error {
	if m.Quantization != QuantizationQDQ {
		return nil
	}
	model, err := graph.Load(data)
	if err != nil {
		return fmt.Errorf("%w: %s: reading the graph: %v", ErrContractMismatch, m.ID, err)
	}
	if err := quantize.CheckTensorRT(model); err != nil {
		return fmt.Errorf("%w: %s declares quantization %s: %v", ErrContractMismatch, m.ID, m.Quantization, err)
	}
	return nil
}

// readCalibration reads, verifies and parses the manifest's calibration
// table, and checks that it names tensors of the graph. It returns nil for a
// manifest without one.
func readCalibration(m Manifest, dir string, model []byte) ([]byte, error) {
	c := m.Calibration
	if c == nil {
		return nil, nil
	}
	path := filepath.Join(dir, c.File)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: reading the calibration table of %s: %w", m.ID, err)
	}
	if got := digest(data); got != c.SHA256 {
		return nil, fmt.Errorf("%w: the calibration table of %s at %s has sha256 %s and its manifest records %s",
			ErrDigestMismatch, m.ID, path, got, c.SHA256)
	}
	table, err := calibration.Parse(data, c.Format)
	if err != nil {
		return nil, fmt.Errorf("onnx: the calibration table of %s: %w", m.ID, err)
	}
	graph, err := onnxgraph.Read(model)
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrContractMismatch, m.ID, err)
	}
	known := map[string]bool{}
	for _, name := range graph.Activations() {
		known[name] = true
	}
	matched := 0
	for name := range table {
		if known[name] {
			matched++
		}
	}
	if matched == 0 {
		return nil, fmt.Errorf("%w: %s: none of the %d tensors of its calibration table is a tensor of the graph, so the table belongs to another model",
			ErrContractMismatch, m.ID, len(table))
	}
	return data, nil
}

// digest is the lowercase hexadecimal SHA-256 of the bytes.
func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// builtSession is what createSession returns.
type builtSession struct {
	session   *ort.DynamicAdvancedSession
	provider  string
	placement Placement
}

// createSession builds the session with the available accelerators in the
// configured order. When the library refuses the graph with them, the first
// is dropped and the build is tried with the rest, down to the CPU alone,
// which RequireAccelerated refuses.
func (r *Runtime) createSession(m Manifest, data, table []byte, plan SessionPlan) (builtSession, error) {
	available := map[string]bool{}
	for _, status := range r.providers {
		available[status.Name] = status.Available
	}
	accelerators := []string{}
	for _, name := range plan.Accelerators {
		if available[name] {
			accelerators = append(accelerators, name)
		}
	}

	var errs []error
	for i := range accelerators {
		attempt := accelerators[i:]
		built, err := r.build(m, data, table, plan, attempt)
		if err == nil {
			return built, nil
		}
		if errors.Is(err, ErrCPUFallback) {
			return builtSession{}, err
		}
		errs = append(errs, fmt.Errorf("%s: %w", strings.Join(attempt, "+"), err))
	}

	if r.cfg.Placement.RequireAccelerated {
		err := checkPlacement(m.ID, plan.Device, nil, Placement{}, nil)
		return builtSession{}, errors.Join(append([]error{err}, errs...)...)
	}
	built, err := r.build(m, data, table, plan, nil)
	if err != nil {
		errs = append(errs, err)
		return builtSession{}, fmt.Errorf("onnx: creating a session for %s: %w", m.ID, errors.Join(errs...))
	}
	return built, nil
}

// build builds one session with the accelerators given in front of the CPU,
// none for the CPU alone, capturing its placement when that is configured and
// not yet known for the model on the device, and applying RequireAccelerated
// to an accelerated one.
func (r *Runtime) build(m Manifest, data, table []byte, plan SessionPlan, accelerators []string) (builtSession, error) {
	if len(accelerators) > 0 {
		acceleratedBuilds.Lock()
		defer acceleratedBuilds.Unlock()
	}

	if containsProvider(accelerators, ProviderTensorRT) {
		if err := prepareTensorRT(plan, table); err != nil {
			return builtSession{}, err
		}
	}

	key := placementKey(m, plan, accelerators)
	r.placementsMu.Lock()
	placement, known := r.placements[key]
	r.placementsMu.Unlock()

	var session *ort.DynamicAdvancedSession
	build := func(verbose bool) func() error {
		return func() error {
			var err error
			session, err = r.newSession(m, data, accelerators, plan, verbose)
			return err
		}
	}
	var err error
	if r.cfg.Placement.Capture && !known {
		placement, err = capturePlacement(build(true))
		if err == nil && placement.Captured {
			r.placementsMu.Lock()
			r.placements[key] = placement.clone()
			r.placementsMu.Unlock()
		}
	} else {
		err = build(false)()
	}
	if err != nil {
		return builtSession{}, err
	}

	if r.cfg.Placement.RequireAccelerated && len(accelerators) > 0 {
		if err := checkPlacement(m.ID, plan.Device, accelerators, placement, r.cfg.Placement.CPUOps); err != nil {
			_ = session.Destroy()
			return builtSession{}, err
		}
	}
	provider := ProviderCPU
	if len(accelerators) > 0 {
		provider = accelerators[0]
	}
	return builtSession{session: session, provider: provider, placement: placement.clone()}, nil
}

// placementKey names one model, with one calibration table, on one device,
// built for one list of accelerators: the unit whose placement is captured
// once.
func placementKey(m Manifest, plan SessionPlan, accelerators []string) string {
	return m.SHA256 + "+" + plan.Calibration + "@" + strconv.Itoa(plan.Device) + "/" + strings.Join(accelerators, "+")
}

func containsProvider(names []string, name string) bool {
	for _, n := range names {
		if n == name {
			return true
		}
	}
	return false
}

// prepareTensorRT creates the model's engine cache directory and writes the
// verified calibration table into it, where TensorRT reads it.
//
// ONNX Runtime creates only the last element of a cache path, so the whole of
// it is created here, private to the process. The table is rewritten only when
// what is there differs from the verified bytes, and is replaced atomically,
// so a concurrent reader sees one table or the other and never half of one.
func prepareTensorRT(plan SessionPlan, table []byte) error {
	if plan.EngineCacheDir == "" {
		return nil
	}
	if err := os.MkdirAll(plan.EngineCacheDir, 0o700); err != nil {
		return fmt.Errorf("onnx: the TensorRT cache of %s: %w", plan.Model, err)
	}
	if plan.CalibrationPath == "" || table == nil {
		return nil
	}
	if current, err := os.ReadFile(plan.CalibrationPath); err == nil && bytes.Equal(current, table) {
		return nil
	}
	tmp, err := os.CreateTemp(plan.EngineCacheDir, ".calibration-*")
	if err != nil {
		return fmt.Errorf("onnx: writing the calibration table of %s: %w", plan.Model, err)
	}
	_, writeErr := tmp.Write(table)
	closeErr := tmp.Close()
	if err := errors.Join(writeErr, closeErr); err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("onnx: writing the calibration table of %s: %w", plan.Model, err)
	}
	if err := os.Rename(tmp.Name(), plan.CalibrationPath); err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("onnx: writing the calibration table of %s: %w", plan.Model, err)
	}
	return nil
}

// newSession builds one session with the given accelerators in front of the
// CPU, at verbose log severity when the placement is being captured.
func (r *Runtime) newSession(m Manifest, data []byte, accelerators []string, plan SessionPlan, verbose bool) (*ort.DynamicAdvancedSession, error) {
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
	if verbose {
		if err := options.SetLogSeverityLevel(ort.LoggingLevelVerbose); err != nil {
			return nil, err
		}
	}
	for _, name := range accelerators {
		if err := appendProvider(options, name, &plan); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	}
	return ort.NewDynamicAdvancedSessionWithONNXData(data, names(m.Inputs), names(m.Outputs), options)
}
