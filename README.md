<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Arandu ONNX

It runs ONNX models through ONNX Runtime and answers with tensors, and it
decides nothing.

This is the shared runtime of the Arandu vision engines —
[`arandu-paddleocr`](https://github.com/hyz-is/arandu-paddleocr),
[`arandu-face`](https://github.com/hyz-is/arandu-face) and
[`arandu-liveness`](https://github.com/hyz-is/arandu-liveness). An engine hands
it tensors and gets tensors back; what those numbers mean is the engine's
business, and what is done about them is the application's.

What it holds:

- **Runtime** — loads the ONNX Runtime shared library, probes the preferred
  execution providers (`cpu`, `coreml`, `cuda`) and falls back to the CPU when
  one is missing.
- **Model manifests** — every model carries its provenance (source, licences,
  commercial use, redistribution, status) and its contract (inputs, outputs,
  precision). `Open` verifies the file's SHA-256 before ONNX Runtime reads it,
  refuses a `blocked` model, and checks the graph against the declared tensors.
- **Sessions** — `Run` takes and returns plain `float32`, `int64` and `uint8`
  tensors, honours the context, is safe for concurrent use, and attaches a
  `ModelInfo` (model id, version, digest, runtime version, precision, provider,
  status) to every result, for the caller to record as evidence.
- **Image helpers** — decode JPEG/PNG upright by EXIF orientation, bilinear
  resize, letterbox, crop, and NCHW normalisation. Pure Go.

## What it is not

- **Not a decision.** There is no threshold, no verdict, no "approved", no
  "match" anywhere in this package. It returns the raw output tensors and the
  identity of the model that produced them.
- **Not a model zoo.** It ships no model weights. A model is opened only
  through a manifest that names its digest and its licences; the engine modules
  carry the manifests of the models they use.
- **Not a server.** Registering it adds no route, no table and no background
  loop.

## Requirements

| | |
| --- | --- |
| Go | 1.26 or later (`go.mod` declares the minimum) |
| cgo | `CGO_ENABLED=1` and a C compiler. The binding, [`github.com/yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go) v1.31.0, is cgo |
| ONNX Runtime | the shared library, **1.26 or later**. The binding requests C API version 26, and an older library refuses it. Development and CI use **1.30.0** |
| Arandu | framework 0.50 or later |

The library is loaded with `dlopen` when the module boots, not at build time,
so building, vetting and testing code that only constructs the module needs a C
compiler and nothing else.

### Installing ONNX Runtime

| OS | install | default `SharedLibraryPath` |
| --- | --- | --- |
| macOS, Apple Silicon | `brew install onnxruntime` | `/opt/homebrew/lib/libonnxruntime.dylib` |
| macOS, Intel | `brew install onnxruntime` | `/usr/local/lib/libonnxruntime.dylib` |
| Linux x64 / arm64 | the release archive, below | `libonnxruntime.so`, resolved by the dynamic loader |
| Windows | the release `.zip`; put `onnxruntime.dll` beside the binary or on `PATH` | `onnxruntime.dll`, resolved by the loader |

On Linux, the same steps CI runs:

```bash
version=1.30.0
arch=x64   # or aarch64
curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-linux-${arch}-${version}.tgz" \
  | tar -xz -C /tmp
sudo cp -P /tmp/onnxruntime-linux-${arch}-${version}/lib/libonnxruntime*.so* /usr/local/lib/
sudo ldconfig
```

A library anywhere else is named with `Config.SharedLibraryPath`. The defaults
are defaults, not requirements.

## Install

```bash
go get github.com/hyz-is/arandu-onnx@v0.1.0
```

## Wire it

An Arandu application registers a module explicitly. There is no service
provider, no container and no discovery, so these are the lines to paste into
`bootstrap/app.go` and there are no others.

The import, with the other module imports:

```go
import (
	onnx "github.com/hyz-is/arandu-onnx"
)
```

The construction, in `Build`, before `k.Register`. Every value comes from the
application's own typed configuration; this package reads no environment
variable:

```go
	onnxModule, err := onnx.New(onnx.Config{
		SharedLibraryPath: cfg.ONNX.Library, // empty: the default above
		IntraOpThreads:    cfg.ONNX.Threads, // 0: ONNX Runtime chooses
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
whether it is loaded.

The engines open their models on its runtime:

```go
	ocrModule, err := paddleocr.New(cfg.OCR, onnxModule.Runtime())
	faceModule, err := face.New(cfg.Face, onnxModule.Runtime())
	livenessModule, err := liveness.New(cfg.Liveness, onnxModule.Runtime())
```

## Configuration

| Field | Default | What it does |
| --- | --- | --- |
| `SharedLibraryPath` | per OS, above | The ONNX Runtime library to load. One per process: a second `Runtime` with a different path is refused |
| `IntraOpThreads` | `0` | Threads inside one operator; 0 lets ONNX Runtime choose (the physical cores). Refused above 1024 |
| `Providers` | `["cpu"]` | Preference order among `cpu`, `coreml`, `cuda`. A provider the library or the machine lacks is skipped and reported by `Runtime.Providers()`, never refused; the CPU is always the last resort |

`New` validates the configuration and returns an error rather than starting
half-wired.

## API overview

### Runtime

```go
rt := onnxModule.Runtime()        // or onnx.NewRuntime(cfg) + rt.Init() outside an application
rt.Version()                      // "1.30.0"
for _, p := range rt.Providers() { // which providers were found, and why one was not
	fmt.Println(p.Name, p.Available, p.Reason)
}
```

`Init` and `Close` are idempotent and safe for concurrent use; a closed runtime
stays closed. The binding keeps its environment in package globals, so one
process holds one library, shared by reference count between `Runtime` values.

### Manifests and sessions

```go
manifest, err := onnx.LoadManifest(manifestJSON) // unknown keys are refused
session, err := rt.Open(manifest, modelDir)      // blocked? digest? contract? then load

result, err := session.Run(ctx, map[string]onnx.Tensor{
	"x": {Shape: []int64{1, 3, 640, 640}, Float32: pixels},
})
scores := result.Outputs["scores"].Float32
evidence := result.Model // ID, Version, SHA256, Runtime, Precision, Provider, Status
```

A manifest:

```json
{
  "id": "example.detector",
  "version": "1.0.0",
  "file": "detector/model.onnx",
  "sha256": "…64 lowercase hex…",
  "source_url": "https://…/model.onnx pinned to a revision",
  "license": "Apache-2.0",
  "model_license": "Apache-2.0",
  "commercial_use": "yes",
  "redistribution": "allowed with notice",
  "modification": "allowed",
  "status": "approved",
  "inputs":  [{"name": "x", "element_type": "float32", "shape": [1, 3, -1, -1]}],
  "outputs": [{"name": "scores", "element_type": "float32", "shape": [1, -1]}],
  "precision": "fp32"
}
```

- `status` is `approved`, `dev-only` (loaded, and `ModelInfo.Status` says so,
  so the evidence of every result carries it) or `blocked` (refused before the
  file is read).
- `-1` is a dynamic dimension.
- `file` is relative to the directory passed to `Open` and may not leave it.
- The licence fields are not interpreted by code. They are what a reviewer read
  before setting `status`, kept beside the file so every result can point back
  to them.

Errors are sentinels, for `errors.Is`: `ErrLibraryUnavailable`,
`ErrNotInitialized`, `ErrClosed`, `ErrModelBlocked`, `ErrDigestMismatch`,
`ErrContractMismatch`, `ErrInvalidManifest`, `ErrInvalidTensor`,
`ErrImageTooLarge`.

### Image helpers

```go
img, err := onnx.DecodeImage(r)                     // JPEG/PNG, EXIF-upright, ≤ 50 MP
boxed, info, err := onnx.Letterbox(img, 640, 640, color.RGBA{114, 114, 114, 255})
x, y := info.ToSource(px, py)                       // back to the original image
tensor, err := onnx.NCHW(boxed, onnx.Normalization{
	Mean: [3]float32{0.5, 0.5, 0.5},
	Std:  [3]float32{0.5, 0.5, 0.5},
})
```

`Resize` and `Crop` complete the set. `DecodeImage` refuses an image above
`MaxImagePixels` before allocating it.

## Concurrency

`Run` is safe for concurrent use. Concurrent calls share one ONNX Runtime
session, whose `Run` is thread-safe, and each call allocates its own input and
output values. A session's reader-writer lock lets runs proceed together and
makes `Close` wait for them.

## Why there is no table, policy or route

This module wraps a runtime and owns no record, so there is nothing for a policy
to decide about; a route would run a model for anybody who sent a tensor.
Inference is reached through the engine modules, behind the controllers and
Grants of the application that installed them. `arandu.mod.toml` declares
`filesystem = true` (it reads model files) and nothing else, and
`tests/Unit/audit_test.go` checks that declaration against what the code calls.

## Limitations

- A model whose weights live in external data files cannot be opened: the
  session is built from the verified bytes in memory.
- Tensor element types are `float32`, `int64` and `uint8`.
- `coreml` and `cuda` are attempted and reported; only the CPU path is
  exercised by the tests and CI.
- Performance depends on the model, the provider and the machine. The engine
  modules publish measured figures for their own models.

## Security

- `Open` hashes the file and compares it with the manifest before ONNX Runtime
  parses a byte of it, so a swapped or truncated model never reaches the native
  parser.
- `DecodeImage` bounds the pixel count before decoding, so an image header
  cannot make the process allocate gigabytes.
- The inputs the engines pass through this package are often personal data
  (faces, identity documents). Nothing here stores, logs or transmits them.

Report vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## Tests

```bash
export GOWORK=off CGO_ENABLED=1
test -z "$(gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go'))"
go build ./...
go vet ./...
go test -race -count=1 ./...
```

The tests write their own models as hand-encoded ONNX protobuf in Go; no model
file is downloaded and none is committed. The tests that load a model need the
ONNX Runtime library at `DefaultSharedLibraryPath()` and skip, saying why, when
it is absent. CI installs ONNX Runtime 1.30.0 so that none of them skips there.

## Versioning and releases

The module follows [Semantic Versioning](https://semver.org/). While the major
version is `v0`, a minor release may break the API, and every break is named in
[UPGRADE.md](UPGRADE.md); CI compares the exported API with the last tag and
fails when a removal is not named there.

A release is a tag `vX.Y.Z` on `main` whose notes are the `## [X.Y.Z] - date`
entry already committed in [CHANGELOG.md](CHANGELOG.md). The release workflow
re-runs the gates on the exact tag archive, audits known vulnerabilities, and
publishes those notes; it never writes to the repository. A published version is
immutable, and a mistake is corrected by another release.

Moving the `onnxruntime_go` binding moves the minimum ONNX Runtime version, and
is always a changelog and upgrade-note line.

## Licence

MIT. See [LICENSE.md](LICENSE.md). Copyright HYZIS - Serviços Digitais Ltda.

ONNX Runtime (MIT, Microsoft) and `onnxruntime_go` (MIT) are dependencies, not
part of this repository; ONNX Runtime is installed separately on the machine
that runs the application.
