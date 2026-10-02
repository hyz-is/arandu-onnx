<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Arandu ONNX

An Arandu package that runs ONNX models through ONNX Runtime and answers with
tensors. It is the shared runtime of KUAA's vision engines — OCR, face,
liveness — and it decides nothing: an engine hands it tensors, gets tensors
back, and what those numbers mean is the engine's business.

What it holds:

- **Runtime** — loads the ONNX Runtime shared library, probes the preferred
  execution providers (`cpu`, `coreml`, `cuda`) and falls back to the CPU when
  one is missing.
- **Model manifests** — every model carries its provenance (source, licences,
  commercial use, status) and its contract (inputs, outputs, precision).
  `Open` verifies the file's SHA-256 before ONNX Runtime reads it, refuses a
  `blocked` model, and checks the graph against the declared tensors.
- **Sessions** — `Run` takes and returns plain `float32`, `int64` and `uint8`
  tensors, honours the context, is safe for concurrent use, and attaches a
  `ModelInfo` (model id, version, digest, runtime version, precision, provider)
  to every result, for the caller to record as evidence.
- **Image helpers** — decode JPEG/PNG upright by EXIF orientation, bilinear
  resize, letterbox, crop, and NCHW normalisation. Pure Go.

## Build requirement

This package links ONNX Runtime through cgo
([`github.com/yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go)
v1.31.0, which loads the library with `dlopen` and requests C API version 26).
An application that registers it:

- builds with `CGO_ENABLED=1` and a C compiler;
- needs the ONNX Runtime shared library, version 1.26 or later, on the machine
  it runs on. Development and the tests use 1.27.1.

The library is not loaded until the module boots, so building, vetting and
testing code that only constructs the module needs neither.

| OS | default `SharedLibraryPath` |
| --- | --- |
| macOS, Apple Silicon | `/opt/homebrew/lib/libonnxruntime.dylib` |
| macOS, Intel | `/usr/local/lib/libonnxruntime.dylib` |
| Linux | `libonnxruntime.so`, resolved by the dynamic loader |
| Windows | `onnxruntime.dll`, resolved by the loader |

## Install

```bash
go get github.com/kuaa-io/arandu-onnx
```

## Wire it

An Arandu application registers a module explicitly. There is no service
provider, no container and no discovery, so these are the lines to paste into
`bootstrap/app.go` and there are no others.

The import, with the other module imports:

```go
import (
	onnx "github.com/kuaa-io/arandu-onnx"
)
```

The construction, in `Build`, before `k.Register`. Every value comes from the
application's own typed configuration; this package reads no environment
variable:

```go
	onnxModule, err := onnx.New(onnx.Config{
		SharedLibraryPath: cfg.ONNX.Library,      // empty: the default above
		IntraOpThreads:    cfg.ONNX.Threads,      // 0: ONNX Runtime chooses
		Providers:         []string{"coreml", "cpu"},
	})
	if err != nil {
		return App{}, err
	}
```

And the registration, inside the `k.Register(...)` call already there:

```go
		onnxModule,
```

`Boot` loads the library and fails the boot, naming the path, when it cannot.
`Close` closes every session and unloads it at shutdown. `Health` reports
whether it is loaded. Registering it adds no route, no table and no background
loop.

The engines open their models on its runtime:

```go
	ocrModule, err := paddleocr.New(cfg.OCR, onnxModule.Runtime())
```

## Run a model

```go
runtime := onnxModule.Runtime()
manifest, err := onnx.LoadManifest(manifestJSON) // the model's manifest.json
session, err := runtime.Open(manifest, modelDir)  // verifies, then loads

result, err := session.Run(ctx, map[string]onnx.Tensor{
	"x": {Shape: []int64{1, 3, 640, 640}, Float32: pixels},
})
scores := result.Outputs["scores"].Float32
evidence := result.Model // ID, Version, SHA256, Runtime, Precision, Provider, Status
```

A manifest:

```json
{
  "id": "paddleocr.det",
  "version": "5.0.0",
  "file": "det/model.onnx",
  "sha256": "…64 lowercase hex…",
  "source_url": "https://…",
  "license": "Apache-2.0",
  "model_license": "Apache-2.0",
  "commercial_use": "yes",
  "redistribution": "allowed with notice",
  "modification": "allowed",
  "status": "approved",
  "inputs":  [{"name": "x", "element_type": "float32", "shape": [1, 3, -1, -1]}],
  "outputs": [{"name": "fetch_name_0", "element_type": "float32", "shape": [1, 1, -1, -1]}],
  "precision": "fp32"
}
```

`status` is `approved`, `dev-only` (loaded, and `ModelInfo.Status` says so) or
`blocked` (refused before the file is read). `-1` is a dynamic dimension.
Unknown keys are refused.

## Configuration

| Field | Default | What it does |
| --- | --- | --- |
| `SharedLibraryPath` | per OS, above | The ONNX Runtime library to load. One per process |
| `IntraOpThreads` | `0` | Threads inside one operator; 0 lets ONNX Runtime choose. Refused above 1024 |
| `Providers` | `["cpu"]` | Preference order among `cpu`, `coreml`, `cuda`. A provider the library lacks is skipped and reported by `Runtime.Providers()`; the CPU is always the last resort |

## Concurrency

`Run` is safe for concurrent use. Concurrent calls share one ONNX Runtime
session, whose `Run` is thread-safe, and each call allocates its own input and
output values. A session's reader-writer lock lets runs proceed together and
makes `Close` wait for them. `Runtime.Init` and `Runtime.Close` are idempotent;
a closed runtime stays closed.

## Why there is no table, policy or route

This repository started from the Arandu package skeleton, whose demo entity —
model, policy, service, views, migration and three routes — was removed. This
module wraps a runtime and owns no record, so there is nothing for a policy to
decide about; a route would run a model for anybody who sent a tensor.
Inference is reached through the engine modules, behind the controllers and
Grants of the application that installed them. `arandu.mod.toml` declares
`filesystem = true` (it reads model files) and nothing else.

## Limitations

- A model whose weights live in external data files cannot be opened: the
  session is built from the verified bytes in memory.
- Tensor element types are `float32`, `int64` and `uint8`.
- `coreml` and `cuda` are attempted and reported, and only the CPU path is
  exercised by the tests.

## Gates

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

The tests that load models skip, saying why, when the library is not on the
machine. Their models are hand-encoded ONNX protobuf written by the tests
themselves.

## Licence

MIT. See [LICENSE.md](LICENSE.md).
