---
name: onnx-package
description: Install, wire and use the Arandu ONNX package (Go, Arandu) in an application. Use when the request is to "install arandu-onnx", "run an ONNX model", "add ONNX Runtime", "go get github.com/hyz-is/arandu-onnx", "wire it into bootstrap/app.go", "the app fails to boot with ONNX Runtime library unavailable", "dlopen failed", "model digest mismatch", "model blocked", "model contract mismatch", "use CoreML", "use CUDA", "write a model manifest", or when a project's go.mod already requires github.com/hyz-is/arandu-onnx. Covers the build requirement, the wiring and where it goes, the three Config fields, manifests, and what each error means.
license: MIT
---

# Using Arandu ONNX

A module that loads ONNX Runtime, opens verified models and runs them,
returning tensors and the `ModelInfo` that produced them. It decides nothing.
It is registered by hand — there is no service provider, no container and no
discovery.

## Build requirement

`CGO_ENABLED=1`, a C compiler, and the ONNX Runtime shared library (1.26 or
later) on the machine that runs the application. On macOS with Homebrew:
`brew install onnxruntime`. On Linux: the release archive from the ONNX
Runtime project, with `libonnxruntime.so` on the loader's path or named in
`Config.SharedLibraryPath`.

## Install

```bash
go get github.com/hyz-is/arandu-onnx@v0.1.0
```

## The wiring, and where it goes

All of it is in `bootstrap/app.go`.

```go
import (
	onnx "github.com/hyz-is/arandu-onnx"
)
```

In `Build`, before `k.Register`, with every value from the application's typed
configuration (the package reads no environment variable):

```go
	onnxModule, err := onnx.New(onnx.Config{
		SharedLibraryPath: cfg.ONNX.Library, // empty: DefaultSharedLibraryPath()
		IntraOpThreads:    cfg.ONNX.Threads, // 0: ONNX Runtime chooses
		Providers:         cfg.ONNX.Providers, // e.g. coreml, cpu
	})
	if err != nil {
		return App{}, err
	}
```

Inside the `k.Register(...)` call:

```go
		onnxModule,
```

Then hand `onnxModule.Runtime()` to each engine module's constructor. Boot loads
the library; shutdown closes every session and unloads it.

## Models

Every model is a file plus a manifest (JSON, unknown keys refused): `id`,
`version`, `file` (relative to the model directory), `sha256`, `source_url`,
`license`, `model_license`, `commercial_use` (`yes|no|unknown`),
`redistribution`, `modification`, `status` (`approved|dev-only|blocked`),
`inputs`/`outputs` (`name`, `element_type` `float32|int64|uint8`, `shape` with
`-1` for dynamic), `precision` (`fp32|fp16|int8`).

```go
manifest, err := onnx.LoadManifest(body)
session, err := runtime.Open(manifest, dir)
result, err := session.Run(ctx, map[string]onnx.Tensor{"x": input})
```

Record `result.Model` with whatever the application derives from
`result.Outputs`.

## Symptoms to recognise

- **Boot fails with `ONNX Runtime library unavailable`.** The library is not at
  the path, cannot be loaded (the loader's message follows, often a missing
  dependency), or is older than the C API version the binding requests.
- **`model digest mismatch`.** The file is not the one the manifest recorded.
  Do not edit the digest to make it pass; find out where the file came from.
- **`model blocked`.** The manifest says the model must not run.
- **`model contract mismatch`.** The graph's inputs or outputs are not the
  manifest's: wrong name, element type, rank or fixed dimension.
- **`invalid tensor`.** An input is missing, extra, of the wrong type, or its
  data does not fill its shape.
- **`Providers()` shows `coreml` or `cuda` unavailable.** The library was built
  without it; the session runs on the CPU and `ModelInfo.Provider` says so.
