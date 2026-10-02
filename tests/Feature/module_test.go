package feature_test

import (
	"testing"

	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// These tests drive the module the way an application does: build it and
// register it on a router.

func TestTheModuleIsRegisteredAsAModule(t *testing.T) {
	t.Parallel()

	m, err := onnx.New(onnx.Config{})
	if err != nil {
		t.Fatalf("building the module: %v", err)
	}
	var module foundation.Module = m
	if module.Name() != "onnx" {
		t.Fatalf("Name() = %q, want onnx: route names and the boot log are keyed by it", module.Name())
	}
}

// TestTheModuleServesNoRoute holds the reason Routes is empty: inference is
// reached through the engines that call it, behind the application's own
// controllers, and a route here would run a model for anybody.
func TestTheModuleServesNoRoute(t *testing.T) {
	t.Parallel()

	m, err := onnx.New(onnx.Config{})
	if err != nil {
		t.Fatalf("building the module: %v", err)
	}
	router := fhttp.NewRouter()
	m.Routes(router.ForModule(m.Name()))
	if routes := router.Routes(); len(routes) != 0 {
		t.Fatalf("the module registered %d routes, want none: %+v", len(routes), routes)
	}
}
