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
  execution providers (`cpu`, `coreml`, `cuda`, `tensorrt`) and falls back,
  provider by provider, to the CPU when one is missing or refuses the graph.
- **TensorRT and INT8** — TensorRT in front of CUDA on NVIDIA cards, with real
  INT8 on Turing tensor cores (the Tesla T4): a float model plus a pinned
  calibration table, verified before TensorRT reads it.
- **Calibration, in Go** — `Runtime.Calibrate` and the
  `arandu-onnx-calibrate` command measure activation ranges over
  representative inputs and write the table in either format TensorRT
  accepts through ONNX Runtime.
- **Placement guard** — the first session of each model on each device is
  built with ONNX Runtime's verbose log captured, so `Session.Placement` says
  which provider got which nodes, and `RequireAccelerated` refuses a session
  whose work fell back to the CPU.
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
go get github.com/hyz-is/arandu-onnx@v0.2.0
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
| `Providers` | `["cpu"]` | Preference order among `cpu`, `coreml`, `cuda`, `tensorrt`. A provider the library or the machine lacks is skipped and reported by `Runtime.Providers()`, never refused; the CPU is always the last resort. A session the library refuses with every available accelerator is tried again without the first, down to the CPU |
| `Device` | `0` | The GPU `cuda` and `tensorrt` sessions opened with `Open` run on. `OpenOn(manifest, dir, device)` picks another per session |
| `CUDA` | ONNX Runtime's defaults | `GPUMemLimit`, `ConvAlgoSearch`, `ConvUseMaxWorkspace`, `ArenaExtendStrategy`, `PreferNHWC`, and `Extra` for any other CUDA option |
| `TensorRT` | off | `CacheDir`, `EngineCache`, `TimingCache`, `FP16`, `INT8`, `MaxWorkspaceSize`, `BuilderOptimizationLevel`, and `Extra` for any other `trt_*` option except those that load a library, write elsewhere or take a raw pointer |
| `Placement` | off | `Capture` the node placement of the first session of each model on each device; `RequireAccelerated` to refuse a session whose work is on the CPU; `CPUOps` the operators allowed there (default `DefaultCPUOps()`) |

`New` validates the configuration and returns an error rather than starting
half-wired.

## NVIDIA GPUs: TensorRT, INT8 and placement

On a Turing card such as the Tesla T4, CUDA gives FP16 tensor cores and
nothing more: a model quantized for the CPU (QOperator: `QLinearConv`,
`QGemm`) falls back to the CPU node by node, and its INT8 never touches the
card. Real INT8 needs TensorRT. This package puts TensorRT in front of CUDA,
CUDA in front of the CPU, and checks where the nodes landed.

### What the machine needs

| | |
| --- | --- |
| ONNX Runtime | the **GPU** build, 1.26 or later (`onnxruntime-linux-x64-gpu-<version>.tgz`), with `libonnxruntime_providers_cuda.so`, `libonnxruntime_providers_tensorrt.so` and `libonnxruntime_providers_shared.so` beside `libonnxruntime.so` |
| CUDA, cuDNN, TensorRT | the versions that ONNX Runtime release was built with (its release notes and the TensorRT provider's requirements table name them), on the loader's path |
| TensorRT major version | **below 11** for INT8 from a calibration table. ONNX Runtime 1.30.0 compiles the dynamic-range path only `#if NV_TENSORRT_MAJOR < 11` (`tensorrt_execution_provider.cc`), because TensorRT 11 drops implicit quantization. With TensorRT 11 only QDQ models run in INT8 |

A provider whose libraries are missing is reported unavailable by
`Runtime.Providers()`, with ONNX Runtime's reason, and the next one is used.

### Wiring a node with two cards

```go
	onnxModule, err := onnx.New(onnx.Config{
		SharedLibraryPath: cfg.ONNX.Library,
		Providers:         []string{onnx.ProviderTensorRT, onnx.ProviderCUDA, onnx.ProviderCPU},
		Device:            0, // Open uses it; OpenOn(m, dir, 1) opens the second card
		CUDA: onnx.CUDAOptions{
			GPUMemLimit:         cfg.ONNX.GPUMemLimit,
			ConvAlgoSearch:      "HEURISTIC",
			ArenaExtendStrategy: "kSameAsRequested",
		},
		TensorRT: onnx.TensorRTOptions{
			CacheDir:    cfg.ONNX.TensorRTCache, // absolute, on a volume, private to the app
			EngineCache: true,
			TimingCache: true,
			FP16:        true,
			INT8:        true,
		},
		Placement: onnx.PlacementOptions{RequireAccelerated: cfg.ONNX.RequireGPU},
	})
```

An engine module opens each model once per card (`runtime.OpenOn(manifest,
dir, 0)` and `runtime.OpenOn(manifest, dir, 1)`) and spreads the work between
the two sessions. Sessions on accelerators are built one at a time in the
process: concurrent GPU session builds on one card have been reported to
crash ONNX Runtime (issue 26610), and a TensorRT build takes the card anyway.

The first build of a TensorRT engine takes minutes, tens of minutes for a large
model; `EngineCache` keeps it in `CacheDir` for the next start. Each model gets
its own directory there, `<id>-<sha256[:16]>[-<table sha256[:16]>]`, because
ONNX Runtime names an engine after the graph's tensor names and not its
weights: two versions of one architecture in one directory would load each
other's engines.

### INT8, two ways

`TensorRT.INT8` lets TensorRT run INT8 where a model can:

- a **QDQ model** (manifest `"precision": "int8"`) carries its scales in the
  graph, and runs in INT8 with no table;
- a **float model with a calibration table** (manifest `"precision": "fp32"`
  and a `calibration` entry) is implicit quantization: TensorRT runs in INT8
  every tensor the table names, and in FP16 (`FP16`) or FP32 the rest. This is
  the form that runs on a T4 with TensorRT 8.6 and 10, where some QDQ exports
  are refused.

A float model with neither runs without INT8. `SessionPlan` (from
`Config.Plan(manifest, device)`) shows exactly what options a session gets,
on any machine.

### Calibration tables

A manifest pins its table like it pins the model:

```json
  "precision": "fp32",
  "calibration": {
    "file": "det/model.calibration.flatbuffers",
    "sha256": "…64 lowercase hex…",
    "format": "onnxruntime",
    "precision": "int8",
    "method": "minmax",
    "source": "arandu-onnx-calibrate, onnxruntime 1.30.0, minmax over 300 document crops (…)"
  }
```

`Open` reads the table, checks its SHA-256, parses it (ONNX Runtime reads the
FlatBuffers format with no verifier, so a malformed table would reach native
code), refuses one that names no tensor of the graph, and only then writes the
verified bytes into the model's cache directory, which is where TensorRT is
pointed. `ModelInfo.Calibration` carries the table's digest on every result of
a TensorRT session.

The two formats, both read by ONNX Runtime's TensorRT provider
(`ReadDynamicRange` in `tensorrt_execution_provider_utils.h`, identical in
1.27.0 and 1.30.0):

| `format` | file | ONNX Runtime option | content |
| --- | --- | --- | --- |
| `onnxruntime` | FlatBuffers `CalTableFlatBuffers.TrtTable` | `trt_int8_use_native_calibration_table=0` | `dict:[KeyValue{key:string, value:string}]`, the value the dynamic range as decimal text. What ONNX Runtime's own calibrator writes, and what arandu-alpr runs on T4 cards |
| `tensorrt` | TensorRT calibration cache text | `trt_int8_use_native_calibration_table=1` | a first line containing `TRT-`, then `name: hex` per tensor, `hex` the IEEE-754 bits of range / 127. ONNX Runtime splits lines at `:`, so names with `:` (TensorFlow's `op:0`) cannot be written in it |

### Producing a table, in Go

```bash
go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-calibrate@v0.2.0 \
  -manifest models/det/manifest.json \
  -images /secure/calibration/det -size 640x640 -letterbox -mean 0,0,0 -std 1,1,1 \
  -method minmax -exclude-prefix /model/model.22/ -exclude boxes,scores \
  -out models/det/model.calibration.flatbuffers
```

It prints the `calibration` entry, digest included. An engine whose
preprocessing is not a plain resize or letterbox calls the API with its own:

```go
result, err := runtime.Calibrate(ctx, manifest, dir, samples, calibration.Options{
	Method:          calibration.MethodMinMax,
	ExcludePrefixes: []string{"/model/model.22/"},
})
table, err := result.Table.MarshalONNXRuntime()
```

`Calibrate` verifies the model as `Open` does, rewrites the verified bytes in
memory so every activation of the top-level graph is also an output, runs that
on the CPU over the samples, and returns one range per float32 tensor:

| method | range | when |
| --- | --- | --- |
| `minmax` (default) | max(\|min\|, \|max\|) over every sample | ONNX Runtime's MinMax calibrator with symmetric ranges, the method of the tables arandu-alpr runs on T4 cards. Clips nothing |
| `percentile` | the `Percentile` (default 99.999) of \|x\|, from a 2048-bin histogram | clips the rarest values |
| `entropy` | the threshold whose 128-level quantization loses least, by KL divergence | NVIDIA's published entropy algorithm; TensorRT's own `EntropyCalibration2` is closed, so the thresholds are close to, not equal to, its own |

The histogram methods read the samples twice, the second time over the exact
range the first found. Tensors whose quantization ruins the result (a head
where pixel boxes and 0–1 scores share one tensor, an attention MatMul) are
left out with `Exclude` and `ExcludePrefixes`, and run in FP16.

The samples decide the table. They have to be what the model sees in
production, in the same preprocessing; random inputs make a table that runs
and reads wrong, and are for tests. For identity verification the samples are
faces and documents: personal data. Calibrate where they are allowed to be,
keep them out of every repository, and describe them in `source` without
naming anybody.

### Where the nodes went

With `Placement.Capture`, the first session of each model on each device is
built with ONNX Runtime's verbose log while file descriptor 2 is redirected
into a pipe (Linux, macOS and the BSDs); the log is parsed for the
`Node placements` section and written back to standard error. `Session.Placement()`
returns nodes per provider (`TensorrtExecutionProvider=3 CUDAExecutionProvider=12
CPUExecutionProvider=4`) and the operators on the CPU.

`RequireAccelerated` refuses the session, with `ErrCPUFallback`, when the work
is not on an accelerator: every accelerator failed and the CPU is left; the
provider asked for first got no node; a node other than the shape arithmetic in
`DefaultCPUOps()` is on the CPU; or the placement could not be captured, since
an unmeasured placement is not a clean one.

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
`ErrImageTooLarge`, `ErrCPUFallback`, and `calibration.ErrInvalidTable`.

`ModelInfo` also carries `Device` (the GPU of a `cuda` or `tensorrt` session)
and `Calibration` (the digest of the table a `tensorrt` session was handed).

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
`filesystem = true` (it reads model files and their tables, and writes the
verified tables into the TensorRT cache directory) and nothing else, and
`tests/Unit/audit_test.go` checks that declaration against what the code calls.

## Limitations

- A model whose weights live in external data files cannot be opened: the
  session is built from the verified bytes in memory.
- Tensor element types are `float32`, `int64` and `uint8`.
- Only the CPU path, and Core ML on macOS, run in the tests here and in CI.
  CUDA and TensorRT are covered by unit tests of the options they are handed
  (`Config.Plan`), of the tables and of the placement parser, and by
  `TestTensorRTRunsTheCalibratedModelInINT8`, which skips, naming why,
  wherever TensorRT is not available. Run it on a GPU node before trusting a
  release there.
- The Linux placement fixtures are written from ONNX Runtime's source, not
  captured on an NVIDIA machine; the macOS ones are captured.
- Calibration runs the top-level graph only: tensors inside `If` and `Loop`
  subgraphs get no range and run at a higher precision. It runs on the CPU and
  reads float32 activations; a float16 model is calibrated from its float32
  export.
- Performance depends on the model, the provider and the machine. The engine
  modules publish measured figures for their own models.

## Security

- `Open` hashes the file and compares it with the manifest before ONNX Runtime
  parses a byte of it, so a swapped or truncated model never reaches the native
  parser.
- `DecodeImage` bounds the pixel count before decoding, so an image header
  cannot make the process allocate gigabytes.
- A calibration table is verified and parsed before TensorRT reads it, and
  TensorRT is pointed only at the verified copy in the model's private cache
  directory (mode 0700). Anybody who can write that directory can swap an
  engine TensorRT will load, so it belongs to the application's user alone.
- The TensorRT options that load a library (`trt_extra_plugin_lib_paths`,
  the engine decryption library), dump files elsewhere or take a raw pointer
  are refused in `TensorRTOptions.Extra`.
- The inputs the engines pass through this package are often personal data
  (faces, identity documents). Nothing here stores, logs or transmits them.
  The placement capture reads ONNX Runtime's log of a session build, which
  names nodes, not inputs.

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
The one test that needs TensorRT skips everywhere TensorRT is not installed, CI
included, and says so. The placement parser reads the logs in
`tests/Unit/testdata/placement/`, whose README says which were captured.

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
