# Changelog

Everything worth knowing about a release of Arandu ONNX is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

## [0.2.0] - 2026-10-19

TensorRT, real INT8 on NVIDIA tensor cores, calibration in Go, and the node
placement report. Every v0.1.0 configuration and manifest keeps working
unchanged.

### Added

- `ProviderTensorRT` ("tensorrt"), named before `ProviderCUDA`, which takes
  the nodes TensorRT leaves. `TensorRTOptions` (`Config.TensorRT`):
  `CacheDir`, `EngineCache`, `TimingCache`, `FP16`, `INT8`,
  `MaxWorkspaceSize`, `BuilderOptimizationLevel` and `Extra`. INT8 runs a QDQ
  model from its own scales, and a float model from its pinned calibration
  table (implicit quantization, TensorRT below 11).
- `CUDAOptions` (`Config.CUDA`): `GPUMemLimit`, `ConvAlgoSearch`,
  `ConvUseMaxWorkspace`, `ArenaExtendStrategy`, `PreferNHWC` and `Extra`.
- `Config.Device` and `Runtime.OpenOn(manifest, dir, device)`: one session per
  GPU. `MaxDevice`.
- `Manifest.Calibration` (`calibration` in JSON, optional): the table's
  `file`, `sha256`, `format` (`onnxruntime` or `tensorrt`), `precision`
  (`int8`), `method` (`minmax`, `percentile`, `entropy` or `external`) and
  `source`. `Open` verifies its digest, parses it, refuses a table that names
  no tensor of the graph, and writes only the verified bytes into the model's
  cache directory for TensorRT.
- `Config.Plan(manifest, device)` and `SessionPlan`: the provider options a
  session is built with, readable on any machine.
- `PlacementOptions` (`Config.Placement`): `Capture` builds the first session
  of each model on each device with ONNX Runtime's verbose log captured from
  file descriptor 2 (Linux, macOS, the BSDs); `RequireAccelerated` refuses with
  `ErrCPUFallback` a session whose work is on the CPU, whose first provider got
  no node, or whose placement could not be captured; `CPUOps`, defaulting to
  `DefaultCPUOps()`. `Placement`, `ParsePlacement`, `Session.Placement` and the
  `LogProvider*` names.
- `ModelInfo.Device` and `ModelInfo.Calibration`.
- `Runtime.Calibrate`, `Samples` and `CalibrationResult`: run a verified
  float32 model on the CPU over representative inputs, every activation
  exposed by rewriting the graph's outputs in memory, and return its table.
- Package `calibration`: `Table` with `MarshalONNXRuntime` (FlatBuffers
  `TrtTable`) and `MarshalTensorRT` (calibration cache text), `Parse`,
  `ParseONNXRuntime`, `ParseTensorRT`, `Detect`; `Collector` with
  `MethodMinMax` (default), `MethodPercentile` and `MethodEntropy`, and
  exclusion by name and prefix.
- Command `cmd/arandu-onnx-calibrate`: calibrates from images, raw float32
  tensors or synthetic inputs, writes the table and prints the manifest
  entry. It is not linked into an application.

### Changed

- When a session cannot be built with every available accelerator, the first
  is dropped and the build is tried with the rest before the CPU alone. In
  v0.1.0 the CPU came straight after the first failure.
- Sessions on accelerators are built one at a time in the process.
- `arandu.mod.toml` says what the package now writes: the verified
  calibration tables, under the TensorRT cache directory the application
  names. `filesystem` was already `true`.

### Requirements

- Unchanged for the CPU and Core ML: Go 1.26, `CGO_ENABLED=1`,
  `github.com/yalue/onnxruntime_go` v1.31.0, ONNX Runtime 1.26 or later.
- For TensorRT: the GPU build of ONNX Runtime with its CUDA and TensorRT
  provider libraries, and the CUDA, cuDNN and TensorRT it was built with;
  TensorRT below 11 for INT8 from a calibration table.
- No new Go dependency.

## [0.1.0] - 2026-10-03

The first public release, as `github.com/hyz-is/arandu-onnx`.

### Added

- `Runtime`: loads the ONNX Runtime shared library named by `Config`
  (`SharedLibraryPath`, `IntraOpThreads`, `Providers`), probes `cpu`, `coreml`
  and `cuda`, and falls back to the CPU. `Init` and `Close` are idempotent and
  safe for concurrent use. One library per process, shared by reference count.
- `Manifest` and `LoadManifest`: model provenance (digest, source, licences,
  commercial use, redistribution, modification, status, precision) and
  contract (inputs, outputs). Unknown keys are refused; `file` may not leave
  the model directory.
- `Runtime.Open`: refuses `blocked`, verifies the SHA-256 before ONNX Runtime
  sees the file, builds the session from the verified bytes, and checks the
  graph against the manifest.
- `Session.Run`: `float32`, `int64` and `uint8` tensors, context honoured
  before, during and after the run, concurrent use, and `ModelInfo` on every
  `Result`.
- Image helpers: `DecodeImage` (JPEG/PNG, EXIF orientation, pixel ceiling),
  `Resize`, `Letterbox`, `Crop`, `NCHW`.
- `Module`: `foundation.Module` with no route and no table, plus `Bootable`,
  `Health` and `Closable` over the runtime.

### Requirements

- Go 1.26, `CGO_ENABLED=1`, `github.com/yalue/onnxruntime_go` v1.31.0.
- ONNX Runtime 1.26 or later (C API 26). CI and release run against 1.30.0.
