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
- **Graph tools, in Go** — `graph` loads, edits and writes ONNX models losslessly;
  `rewrite.Col2ImToConvTranspose` replaces `Col2Im`, which TensorRT lacks, with
  an exact grouped `ConvTranspose`; `rewrite.UpgradeOpset` raises an old export
  (opset 7 and later) to opset 13, rewriting every operator that changed;
  `rewrite.InitializersOutOfInputs` and `rewrite.DropoutToIdentity` make an
  old exporter's weights constants and its Dropout an Identity;
  `quantize.QDQ` exports symmetric INT8 QDQ from a calibration table and
  `quantize.CheckTensorRT` checks any model against TensorRT's QDQ rules. The
  `arandu-onnx-rewrite` and `arandu-onnx-quantize` commands run them, verify
  the result on the CPU and print the new file's manifest, with the file it
  was derived from.

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
go get github.com/hyz-is/arandu-onnx@v0.4.0
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

- a **QDQ model** (manifest `"precision": "int8"`, and `"quantization": "qdq"`
  for one `Open` checks is in the form TensorRT runs) carries its scales in
  the graph, and runs in INT8 with no table. It is the only INT8 TensorRT 11
  runs; `arandu-onnx-quantize` writes one from a calibration table (below);
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
go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-calibrate@v0.4.0 \
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

## Graph tools: rewrites and the QDQ export

Three things stand between some float models and a T4 running them entirely
in TensorRT: an operator neither TensorRT nor the CUDA provider has, INT8
that TensorRT 11 only takes as QDQ, and old exports that QDQ cannot be
written into (an opset below 13, weights listed as graph inputs, Dropout).
All are fixed by editing the graph, in Go, with three packages and two
commands that are not linked into an application.

| package | what it is |
| --- | --- |
| `graph` | `Load` and `Model.Marshal`: an ONNX model decoded into nodes, attributes, initializers and value info a rewrite can change. Every field it does not interpret (doc strings, metadata, functions, subgraphs, external-data references, quantization annotations) is kept as the bytes it read; a model loaded and marshalled unchanged is the same bytes again. Helpers find constants (initializers and `Constant` nodes), producers, consumers and recorded types, hand out fresh names, and check the topological order |
| `rewrite` | `Col2ImToConvTranspose`, `UpgradeOpset`, `InitializersOutOfInputs`, `DropoutToIdentity` |
| `quantize` | `QDQ` and `CheckTensorRT` |

### Col2Im without Col2Im

`Col2Im` (opset 18, what exporters make of `torch.nn.Fold`) has no kernel in
TensorRT or in ONNX Runtime's CUDA provider, so the graph is split between
the card and the CPU. `rewrite.Col2ImToConvTranspose` replaces each node with
an exact equivalent:

```
Col2Im(x [N, C·K, L], image_shape, block_shape)            K = ∏ block_shape
  ≡ Reshape(x, [0, C·K, L1, …, Lr])                         L = ∏ Li
    ConvTranspose(group = C, kernel_shape = block_shape,
                  the same strides, pads and dilations,
                  output_padding = the rows and columns no block reaches,
                  W [C·K, 1, block_shape…] one-hot: channel c·K + j has a 1 at offset j)
```

`image_shape` and `block_shape` have to be constants and the input's
dimension 1 known (value info, or `Col2ImOptions.InputChannels`); anything
else is refused with `rewrite.ErrUnsupported` and the model is left as it
was. The constants only `Col2Im` read are removed with it.

```bash
go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-rewrite@v0.4.0 \
  -manifest models/example/manifest.json -models models \
  -out models/example/model.convtranspose.onnx
```

The command reads the model through its manifest (digest checked), runs the
original and the rewritten model on the CPU over `-verify` synthetic inputs
(3 by default), and writes nothing when an output differs by more than
`-tolerance` (1e-4). It writes the model, a `.report.json` beside it (nodes
rewritten with every attribute, operator counts before and after, the
measured difference) and prints the manifest of the new file. A calibration
entry of the source is dropped: it pins the original file.

### Old exports: opset 7 to 13

QDQ needs opset 13 (per-channel `DequantizeLinear`), and some published
exports are far older, opset 7 at IR 3. `rewrite.UpgradeOpset` raises the default-domain opset to 13 and
rewrites each node whose operator changed signature or meaning in between,
from the ONNX operator changelog:

| operator (changed at) | what the upgrade writes |
| --- | --- |
| `BatchNormalization` (9) | `spatial` removed; `spatial = 0` refused |
| `Slice` (10), `TopK` (10) | `starts`, `ends`, `axes` and `k` become int64 inputs |
| `Clip` (11), `Pad` (11) | `min`, `max`, `pads` and `value` become inputs (float32 data only) |
| `Resize`-10 (11), `Upsample` (deprecated) | `Resize`-13 with an empty `roi`, `coordinate_transformation_mode` `asymmetric` and, for `nearest`, `nearest_mode` `floor` (scales ≥ 1) or `ceil` (scales ≤ 1), which is what opset 10 computed; mixed or computed nearest scales are refused |
| `Dropout` (12) | `ratio` becomes an input; a mask read below opset 10 is refused |
| `Squeeze`, `Unsqueeze`, `ReduceSum`, `Split` (13) | `axes` and `split` become int64 inputs |
| `Softmax`, `LogSoftmax`, `Hardmax` (13) | kept, axis written, when the axis is the last dimension; otherwise `Flatten` → operator along axis 1 → `Reshape` to the input's shape, the 2-D coercion of opset 1 and 11 exactly |

Every other standard operator has to be on the list of those that compute
the same at 13 (new types, negative axes, optional inputs, attributes whose
default is the old behaviour); `Conv` and `LpPool` with `auto_pad` SAME and a
stride above 1, and `ConvTranspose` with any SAME padding, are refused below
11, where SAME meant something else. An operator on neither list, a model
below opset 7, local functions or subgraphs are refused with
`rewrite.ErrUnsupported`, the model unchanged. The IR version becomes 7.

```bash
go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-rewrite@v0.4.0 \
  -manifest models/example/manifest.json -models models -opset 13 \
  -shape 1,3,48,320 -tolerance 1e-5 -out models/example/model.opset13.onnx
```

Exports this old often hold their weights in `Constant` nodes rather than
initializers; `quantize.QDQ` reads those as weights.

### Weights listed as inputs, and Dropout

Exporters written for IR 3 list every initializer as a graph input too, so
the weights are inputs a caller could override and nothing treats them as
constants: `quantize.QDQ` skips every node. `rewrite.InitializersOutOfInputs`
drops the graph inputs an initializer of the same name backs (IR 4 allows
it, and the IR is raised to 4 when below), and the weights become constants.
`rewrite.DropoutToIdentity` replaces every `Dropout` with an `Identity` of
its data, which is what it computes in inference, keeping the node's name
and output; a training mode that is not the constant false, or a mask that
something reads, is refused. Without the Dropout, a model at opset 10 to 12
is one `quantize.QDQ` raises to 13 itself.

```bash
go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-rewrite@v0.4.0 \
  -manifest models/example/manifest.json -models models \
  -initializers-out-of-inputs -dropout-to-identity \
  -out models/example/model.inference.onnx
```

The rewrite command runs, in this order, `-initializers-out-of-inputs`,
`-dropout-to-identity`, `-opset 13` and the Col2Im rewrite (`-col2im`, on by
default, nothing without a `Col2Im`), and fails when none changed anything.
The manifest it prints names the source in `derived_from` (its id, its
digest and what was done), as does the manifest of a QDQ export.

### QDQ INT8 from a calibration table

`quantize.QDQ` turns a float model and its `calibration.Table` into explicit
quantization in the form TensorRT 10 accepts and TensorRT 11 requires:

| what | how |
| --- | --- |
| activations | `QuantizeLinear` → `DequantizeLinear` per tensor, scale = range / 127, int8 zero point 0; one pair per tensor, shared by the quantized nodes that read it, while float nodes keep the float tensor |
| weights | quantized here into int8 initializers, per output channel (axis 0 for `Conv`, 1 for `ConvTranspose` and `MatMul`, by `transB` for `Gemm`), scale = max\|w\| / 127, rounded half to even into [-127, 127], read by a `DequantizeLinear` with its `axis` and int8 zero points 0. A weight is a constant float tensor: an initializer that is not also a graph input, or the value of a `Constant` node. The float initializers and `Constant` nodes nothing reads any more are removed |
| biases | left float32, with nothing in front of them: TensorRT refuses a dequantized bias and fuses it itself. ONNX Runtime's `WeightBiasQuantization` (graph optimization level 1, which every session gets before TensorRT is handed its nodes) rewrites the one-dimensional bias of a quantized `Conv`, `ConvTranspose` or `Gemm` — or of a `MatMul` it fuses with the `Add` after it — whose output goes straight into a `QuantizeLinear` into an int32 `DequantizeLinear` that TensorRT cannot run, splitting the TensorRT graph on the CPU. So such a bias is kept out of its reach: a `Conv` or `ConvTranspose` bias becomes an `Add` after the node, a `Gemm` or `MatMul` bias is reshaped to `[1, N]`, and `Report.DetachedBiases` lists them |
| which nodes | `conv` (default): `Conv`, `ConvTranspose`. `conv-matmul`: also `MatMul` and `Gemm` with a constant 2-D weight. A product of two activations (attention) is never quantized. `ExcludeNodes`, `ExcludePrefixes` and `ExcludeOpTypes` keep heads, Softmax paths or LayerNorm neighbours in float, and a node whose input the table leaves out stays float |
| opset | per-channel `DequantizeLinear` is opset 13. A model at 10 to 12 is raised when none of the operators whose meaning changed before 13 (Clip, Pad, Resize, Dropout, Softmax, LogSoftmax, Hardmax, Split, Squeeze, Unsqueeze, ReduceSum) is in it, and refused otherwise, naming `rewrite.UpgradeOpset`, which raises any model from opset 7 |

An already quantized model, a table of another model, weights in external
data, or options that leave nothing to quantize are refused with
`quantize.ErrRefused`, and the model is left unchanged. The `Report` lists
every node quantized and skipped (with the reason), every scale, and
`ScalesSHA256`, a digest of all of them.

`quantize.CheckTensorRT` checks any model, this package's or a downloaded
one, against the same rules: constant positive scales, int8 zero points of
0, per-channel scales matching their axis, every `DequantizeLinear` reading
an int8 constant or a `QuantizeLinear`, none reaching a bias (input 2 of a
`Conv`, `ConvTranspose` or `Gemm`, or the `Add` after a `MatMul`, through
reshapes and casts included), no bias ONNX Runtime would quantize as above,
no QOperator (`QLinearConv`) or dynamic (`MatMulInteger`) node.

```bash
go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-quantize@v0.4.0 \
  -manifest models/det/manifest.json \
  -mode conv -exclude-prefix /model/model.22/ \
  -out models/det/model.qdq.onnx
```

The table is the manifest's `calibration` entry (its digest checked) or
`-table`. The command checks the export with `CheckTensorRT`, runs it against
the float model on the CPU over synthetic inputs (`-low`, `-high`, `-verify`),
holds the difference to `-tolerance` (a fraction of the largest float output)
when it is given, writes the model and its `.report.json`, and prints its
manifest:

```json
  "precision": "int8",
  "quantization": "qdq"
```

A manifest that declares `"quantization": "qdq"` is opened only when
`CheckTensorRT` accepts the graph, so an asymmetric or QOperator file pinned
as QDQ is refused by `Open` (`ErrContractMismatch`) instead of being found
on a GPU node. Without the key an int8 manifest behaves as in v0.2.0.

The difference on synthetic inputs bounds the rounding; it is not the
model's accuracy in INT8, which is measured on real samples against FP16 on
the card that runs it.

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
- `quantization` is optional: `qdq`, with precision `int8`, declares the
  explicit QDQ form TensorRT runs, and `Open` refuses a graph
  `quantize.CheckTensorRT` does not accept.
- `derived_from` is optional: the source of a rewritten or quantized file,
  `{"id": …, "sha256": …, "transform": …}`, as the commands print it, so a
  shipped file can be followed back to the one its publisher released. `Open`
  does not read it.

Errors are sentinels, for `errors.Is`: `ErrLibraryUnavailable`,
`ErrNotInitialized`, `ErrClosed`, `ErrModelBlocked`, `ErrDigestMismatch`,
`ErrContractMismatch`, `ErrInvalidManifest`, `ErrInvalidTensor`,
`ErrImageTooLarge`, `ErrCPUFallback`, `calibration.ErrInvalidTable`,
`graph.ErrMalformed`, `graph.ErrUnreadable`, `rewrite.ErrUnsupported`,
`quantize.ErrRefused` and `quantize.ErrNotTensorRTQDQ`.

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
- The graph tools edit the top-level graph only: nodes inside `If`, `Loop` and
  `Scan` bodies and local functions are kept as they were read, never
  rewritten or quantized. They read weights from the file, not from external
  data, and the QDQ export refuses a model whose weights are external.
- The opset upgrade reads opset 7 and later and writes 13 only, and handles
  the standard operators it enumerates; anything else is refused rather than
  guessed. Its Softmax rank and Clip and Pad element types come from a small
  inference over the graph, and where that cannot tell, the upgrade takes the
  form that does not depend on it, or refuses.
- The QDQ export and the rewrites are verified on the ONNX Runtime CPU
  provider only. The TensorRT rules `CheckTensorRT` applies are those of the
  TensorRT 10 documentation; that TensorRT builds an INT8 engine from an export,
  its speed and its accuracy against FP16 are measured on a GPU node, not here.
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
- The rewrite and quantize commands read a model only through its manifest,
  with its digest checked before it is decoded, and a calibration table from
  the manifest only with its digest checked. What they write is a new file with
  its own digest in the manifest they print; the source is never modified.
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
file is downloaded and none is committed. Two checks also rewrite real exports
the machine provides, named only by test variables
(`ARANDU_ONNX_OPSET_UPGRADE_MODEL` and `_SAMPLES`,
`ARANDU_ONNX_WEIGHT_INPUTS_MODEL` and `_SAMPLES`), and skip when they are not
set. The tests that load a model need the
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
