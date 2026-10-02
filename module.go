// Package onnx is an Arandu module that runs ONNX models through ONNX Runtime
// and answers with tensors.
//
// It is the runtime the vision engines build on. It owns no table, serves no
// route and decides nothing: an engine hands it tensors and gets tensors back,
// and what those numbers mean is the engine's business.
//
// An application builds it once in bootstrap/app.go and registers it there.
// There is no service provider, no container and no discovery.
package onnx

import (
	"context"

	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"
)

// Module is what the application registers, and what hands the engines their
// Runtime.
//
// Besides foundation.Module it implements foundation.Bootable, which loads the
// library when the application boots, foundation.Health, which reports whether
// it is loaded, and foundation.Closable, which closes every session and unloads
// it at shutdown.
type Module struct {
	runtime *Runtime
}

// Compile-time proof that the module honors the contracts it claims.
var (
	_ foundation.Module   = (*Module)(nil)
	_ foundation.Bootable = (*Module)(nil)
	_ foundation.Health   = (*Module)(nil)
	_ foundation.Closable = (*Module)(nil)
)

// New returns the module, or the reason it cannot be built.
//
// It returns an error rather than panicking or carrying on, because everything
// it refuses is a wiring mistake, and a wiring mistake found at boot costs one
// restart.
//
// It loads nothing. The library is loaded by Boot, so building the module in a
// test or a command that never boots it needs no ONNX Runtime on the machine.
func New(cfg Config) (*Module, error) {
	runtime, err := NewRuntime(cfg)
	if err != nil {
		return nil, err
	}
	return &Module{runtime: runtime}, nil
}

// Runtime is the runtime the engine modules open their sessions on. The
// application passes it to each engine's constructor.
func (m *Module) Runtime() *Runtime { return m.runtime }

// Name is the module identifier: a lowercase slug, stable, no spaces.
func (m *Module) Name() string { return "onnx" }

// Routes registers nothing.
//
// Inference is reached through the engine modules that call it, each behind
// the controller and the Grant of the application that installed it. A route
// here would run a model for anybody who sent a tensor.
func (m *Module) Routes(*fhttp.Router) {}

// Boot loads the shared library and probes the configured providers. A library
// that cannot be loaded stops the boot, with the path in the error.
func (m *Module) Boot(context.Context) error { return m.runtime.Init() }

// Health reports whether the runtime is loaded and not closed.
func (m *Module) Health(context.Context) error { return m.runtime.Health() }

// Close closes every session opened on the runtime and unloads the library.
func (m *Module) Close(context.Context) error { return m.runtime.Close() }
