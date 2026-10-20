# Changelog

Everything worth knowing about a release of Arandu ONNX is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

## [0.4.0] - 2026-10-20

Old exports into QDQ INT8, and QDQ exports that stay whole on TensorRT
behind ONNX Runtime: an opset upgrade from 7 to 13; the rewrites that turn
an IR 3 exporter's weights into constants and its Dropout into an Identity;
and biases kept out of the reach of ONNX Runtime's bias quantization, which
split v0.3.0 exports between TensorRT and the CPU. Configurations and call
sites keep working unchanged; QDQ manifests written by v0.3.0 may now be
refused at `Open` (see Changed).

### Added

- `rewrite.UpgradeOpset` with `OpsetOptions`, `OpsetReport`,
  `OpsetConversion`, `OpsetUpgradeTarget` (13) and `OpsetUpgradeMinimum`
  (7): raises the default-domain opset to 13 and rewrites, from the ONNX
  operator changelog, every node whose operator changed in between
  (BatchNormalization's `spatial`; Slice, TopK, Clip, Pad, Dropout, Squeeze,
  Unsqueeze, ReduceSum and Split attributes to inputs; Resize-10 and
  Upsample to Resize-13 with the opset 10 coordinates and rounding written
  out; Softmax, LogSoftmax and Hardmax kept along their last axis or
  rewritten as Flatten, the operator and Reshape, the old 2-D coercion).
  Operators it does not enumerate, SAME padding that changed meaning at 11,
  nearest resizes whose rounding cannot be written, opsets below 7, local
  functions and subgraphs are refused with `ErrUnsupported`, the model
  unchanged.
- `rewrite.InitializersOutOfInputs` and `InitializersReport`: drops the
  graph inputs an initializer of the same name backs, raising IR to 4 when
  below, so the weights are constants. `rewrite.DropoutToIdentity` and
  `DropoutReport`: replaces every Dropout with an Identity of its data,
  refusing a training mode that is not the constant false or a mask that is
  read.
- `cmd/arandu-onnx-rewrite`: `-opset`, `-initializers-out-of-inputs`,
  `-dropout-to-identity` and `-col2im` (default on), applied in that order
  after the manifest's digest is checked; the report has `initializers`,
  `dropout` and `opset` sections beside `rewrite`.
- `Manifest.DerivedFrom` (`derived_from` in JSON, optional) and
  `Derivation`: the id and digest of the file a rewrite or a quantization
  was made from, and what was done. `arandu-onnx-rewrite` and
  `arandu-onnx-quantize` print it in the manifest of the file they write.
- `quantize.Report.DetachedBiases`: the biases QDQ moved out of the reach of
  ONNX Runtime's bias quantization.

### Changed

- `quantize.QDQ` keeps every bias float32 on the runtime as well as in the
  file. ONNX Runtime's `WeightBiasQuantization`, a level-1 graph transformer
  every session runs before TensorRT is handed its nodes, rewrites the
  one-dimensional bias of a quantized Conv, ConvTranspose or Gemm (or of a
  MatMul its `MatMulAddFusion` joins to the Add after it) whose output goes
  straight into a QuantizeLinear into an int32 DequantizeLinear, which
  TensorRT does not run: each such layer went to the CPU and cut the
  TensorRT graph in pieces. QDQ now writes a Conv or ConvTranspose bias as
  an Add after the node and a Gemm or MatMul bias as `[1, N]`, which compute
  the same and which the transformer does not take. The scales are
  unchanged.
- `quantize.CheckTensorRT` refuses, besides what it refused before, a
  DequantizeLinear whose output reaches a bias through Reshape, Squeeze,
  Unsqueeze, Flatten, Transpose, Cast, Expand or Identity (input 2 of a
  Conv, ConvTranspose or Gemm, and the Add after a MatMul or Gemm), a
  DequantizeLinear that reads neither an int8 constant nor a QuantizeLinear,
  and a bias ONNX Runtime would quantize as above. A manifest declaring
  `"quantization": "qdq"` whose file v0.3.0 exported with such a bias is
  therefore refused at `Open`; export it again.
- `quantize.QDQ` reads a weight from a `Constant` node as well as from an
  initializer, and removes the node when nothing else reads it. A node whose
  weight is an initializer listed as a graph input is skipped with a reason
  naming `rewrite.InitializersOutOfInputs`, and the refusal of a model below
  opset 13 names `rewrite.UpgradeOpset`.
- `arandu-onnx-rewrite` no longer fails on a model without `Col2Im` when
  another rewrite was asked for; it fails when none of the rewrites asked
  for changed anything.
- Calibration leaves out the tensors that do not depend on the inputs: the
  outputs of `Constant` nodes and of nodes that read only constants. They
  have no range to calibrate, and their scalars could not be returned.

### Fixed

- `Session.Run` no longer crashes the process when ONNX Runtime's Go binding
  cannot return an output (a scalar or an empty tensor): the run returns the
  binding's error. Calibrating a model whose scalar `Constant` values were
  taken for activations ended in that crash.

### Requirements

- Unchanged: Go 1.26, `CGO_ENABLED=1`, `github.com/yalue/onnxruntime_go`
  v1.31.0, ONNX Runtime 1.26 or later. No new Go dependency.

## [0.3.0] - 2026-10-19

Graph tools in Go: a lossless ONNX model editor, the Col2Im rewrite that lets
TensorRT run a model with Col2Im whole, and the QDQ INT8 export TensorRT 11 requires. Every
v0.2.0 configuration, manifest and call site keeps working unchanged.

### Added

- Package `graph`: `Load` and `Model.Marshal` decode an ONNX model into
  `Model`, `Graph`, `Node`, `Attribute`, `Tensor` and `ValueInfo` values and
  write it again, keeping every field they do not interpret as the bytes it
  was read as; a model written in ascending field order, as exporters write
  it, round-trips byte for byte (a 117 MB export included).
  `Graph.Initializer`, `Constant` (initializers and `Constant` nodes),
  `Producer`, `Consumers`, `TypeOf`, `RemoveInitializer`, `Used`,
  `CheckTopological`, `OpCounts`; `Namer`; `Model.Opset` and `SetOpset`;
  tensor constructors and readers for float32, int64 and int8. Errors
  `ErrMalformed` and `ErrUnreadable`. No new dependency: it is a hand-written
  codec on the field numbers of onnx.proto.
- Package `rewrite`: `Col2ImToConvTranspose` replaces every `Col2Im` with
  `Reshape` and a grouped `ConvTranspose` with one-hot weights, the same
  strides, pads and dilations, and `output_padding` for what no block
  reaches. `Col2ImOptions.InputChannels` for a graph that does not record an
  input's channels; `Col2ImReport`; `ErrUnsupported`, with the model left
  unchanged.
- Package `quantize`: `QDQ` exports a float model and a `calibration.Table`
  as symmetric INT8 QDQ (activations per tensor, scale range/127; weights per
  output channel into int8 initializers; int8 zero points of 0; biases in
  float), in `ModeConv` (default) or `ModeConvMatMul`, with exclusions by node
  name, prefix and operator, raising opset 10-12 to 13 only when no operator
  changed meaning in between. `Report` with `ScalesSHA256`. `CheckTensorRT`
  checks any model against TensorRT's QDQ rules. `ErrRefused`,
  `ErrNotTensorRTQDQ`, `MinimumOpset`.
- `Manifest.Quantization` (`quantization` in JSON, optional) and
  `QuantizationQDQ` (`"qdq"`), valid with precision `int8`: `Open` refuses,
  with `ErrContractMismatch`, a model so declared that `CheckTensorRT` does
  not accept.
- Command `cmd/arandu-onnx-rewrite`: rewrites Col2Im through the model's
  manifest, verifies the result against the original on the CPU (`-verify`,
  `-tolerance`), writes the model and a JSON report, prints its manifest.
- Command `cmd/arandu-onnx-quantize`: exports QDQ from the manifest's pinned
  table or `-table`, checks it with `CheckTensorRT`, compares it with the float
  model on the CPU, writes the model and a JSON report, prints its manifest
  (precision `int8`, quantization `qdq`).

### Changed

- `.gitignore` ignores `tmp/`, where local runs of the commands and the gate
  logs go.

### Requirements

- Unchanged: Go 1.26, `CGO_ENABLED=1`, `github.com/yalue/onnxruntime_go`
  v1.31.0, ONNX Runtime 1.26 or later. The new packages are pure Go; the
  commands need ONNX Runtime only to verify (`-verify 0` skips it).
- No new Go dependency.

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
