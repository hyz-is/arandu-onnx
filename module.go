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
	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"
)

// Module is what the application registers.
type Module struct {
	cfg Config
}

// Compile-time proof that the module honors the contract it claims.
var _ foundation.Module = (*Module)(nil)

// New returns the module, or the reason it cannot be built.
//
// It returns an error rather than panicking or carrying on, because everything
// it refuses is a wiring mistake, and a wiring mistake found at boot costs one
// restart.
func New(cfg Config) (*Module, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Module{cfg: cfg.withDefaults()}, nil
}

// Name is the module identifier: a lowercase slug, stable, no spaces.
func (m *Module) Name() string { return "onnx" }

// Routes registers nothing.
//
// Inference is reached through the engine modules that call it, each behind
// the controller and the Grant of the application that installed it. A route
// here would run a model for anybody who sent a tensor.
func (m *Module) Routes(*fhttp.Router) {}
