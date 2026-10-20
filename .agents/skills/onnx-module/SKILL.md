---
name: onnx-module
description: Change this Arandu package's runtime, manifests, sessions or image helpers. Use when the request is to "support another element type", "add a provider", "add TensorRT", "add a manifest field", "load a model by path", "skip the digest check", "add a threshold", "return a decision", "add an image helper", "resize differently", "add a route", "rewrite Col2Im", "upgrade the opset", "the model is opset 7", "the weights are graph inputs", "remove Dropout", "convert the model for TensorRT", "edit the ONNX graph", "quantize to QDQ", "export INT8 for TensorRT 11", "check a QDQ model", or when a change touches runtime.go, manifest.go, session.go, tensor.go, image.go, exif.go, config.go, module.go, graph/, rewrite/, quantize/ or the commands in cmd/ Covers what may and may not change, and the tests that hold each property.
license: MIT
---

# Changing the runtime

## Refused outright

- **A decision.** No threshold, verdict, "is live", "matches" or score cut-off
  in this package. It returns tensors and `ModelInfo`.
- **A model without a manifest,** or a way around the digest check, the
  blocked status or the contract check in `Runtime.Open`.
- **Python,** anywhere: no export script, no fixture generator, no test helper.
- **An environment variable** read by the package. Settings are `Config` fields.
- **A route.** Engines are reached through the application's controllers.

## Element types

`Tensor` holds `float32`, `int64` and `uint8`. Adding one is: the field on
`Tensor`, the case in `ElementType`, `newValue`, `fromValue`, `sameElementType`
and `validateSpecs`, the constant, a hand-encoded model in
`tests/Feature/model_fixture_test.go` that takes or returns it, and a changelog
line.

## Providers

A provider is a name in `config.go`, a case in `appendProvider`
(`providers.go`), its options in `Config.plan`, its log name in
`logProviderNames` (`placement.go`), and a line in the README table. It must
fail soft: `probeProviders` records why it is unavailable and `createSession`
(`open.go`) drops the first accelerator and retries, down to the CPU. A
provider that can stop the application from starting is refused;
`RequireAccelerated` refuses a session at `Open`, never the boot.

Provider options are computed in `Config.Plan`, which reads no file and loads
no library, so every option a GPU session gets is unit-tested in
`tests/Unit/plan_test.go` on a machine with no GPU. A new option is a field,
a line in `plan`, a row in that test, and a key in `managedCUDAOptions` or
`managedTensorRTOptions` so `Extra` cannot set it twice. A TensorRT option
that loads a library, writes outside `CacheDir` or takes a raw pointer goes in
`refusedTensorRTOptions`, not in `Extra`.

### TensorRT, INT8 and calibration

- TensorRT is only ever pointed at the **verified copy** of a calibration
  table, written by `prepareTensorRT` into the model's own cache directory
  after `readCalibration` checked its digest, parsed it and matched its names
  against the graph. Pointing `trt_int8_calibration_table_name` at the
  manifest's file directly is refused: what was verified is what runs.
- The engine cache directory is named by the model's and the table's digests.
  ONNX Runtime keys engines by tensor names, not weights; a shared directory
  loads the wrong engine silently.
- The table formats are ONNX Runtime's (`ReadDynamicRange` in
  `core/providers/tensorrt/tensorrt_execution_provider_utils.h`). Before
  changing `calibration/flatbuffers.go` or `calibration/tensorrt.go`, read
  that function at the ONNX Runtime version supported, and keep the golden
  layouts in `tests/Unit/calibration_table_test.go` written by hand.
- Implicit INT8 from a table exists only for TensorRT below 11
  (`#if NV_TENSORRT_MAJOR < 11` in the provider). Say so wherever a version
  moves.
- Calibration is Go: `Runtime.Calibrate` rewrites the verified graph's outputs
  in memory (`internal/onnxgraph`) and runs it on the CPU. No Python
  calibrator, no exported augmented model on disk, no samples in the
  repository: tests use the hand-encoded models and hand-computed ranges.

### Placement

`capturePlacement` redirects file descriptor 2 for one session build; it is
serialised and writes everything back. `ParsePlacement` reads ONNX Runtime's
`Node placements` section; its fixtures are in `tests/Unit/testdata/placement`
and their README says which were captured. Replace a hand-written Linux
fixture with a captured one when an NVIDIA machine is at hand; never
"fix" a parser test by editing a captured fixture.

## Manifest fields

Manifests live beside models in other repositories, and unknown keys are
refused, so a new required field breaks every existing manifest. Add it
optional first, or say so as a breaking change in `UPGRADE.md`.

## Image helpers

Pure Go on the standard library. Each helper has golden values in
`tests/Unit/image_test.go`, computed by hand in the test, not by running the
helper. Keep the OpenCV pixel-centre convention in `Resize`: engines port
preprocessing from references that use it.

## Concurrency

`Session.Run` holds the session's read lock; `Close` the write lock.
`Runtime.Open` holds the runtime's read lock for the whole load so the
environment cannot be destroyed under it. Any change here runs
`go test -race ./...` with the library present, and reads the skip lines.

## Graph tools: graph, rewrite, quantize

- **`graph` is lossless.** It decodes the fields a rewrite needs and keeps every
  other field as the bytes it read. A new field it interprets is decoded,
  re-encoded in ascending field order, and covered by the byte-for-byte
  round-trip test in `tests/Feature/graph_test.go`; run the probe on a real
  export too (a real export round-trips identically). Never drop a field to make a
  rewrite simpler. The package reads no file: the commands do.
- **A rewrite is all or nothing.** Plan every node first, refuse with the
  package's sentinel and leave the model unchanged, and only then edit; end
  with `Graph.CheckTopological`. A refusal test asserts the marshalled bytes
  are the input's.
- **A rewrite is proven numerically.** A test runs the original and the
  rewritten graph on ONNX Runtime CPU (hand-encoded fixtures, random synthetic
  inputs) and pins at least one hand-computed result, because ONNX Runtime may
  compute both sides with the same kernel (its ConvTranspose is GEMM plus
  col2im, which is why the Col2Im difference is exactly 0).
- **QDQ is TensorRT's form and nothing else.** Symmetric int8, zero point 0,
  per-tensor activations, per-channel weights with `axis`, float biases with no
  DequantizeLinear. Any change keeps `CheckTensorRT` and `QDQ` agreeing, and
  `assertTensorRTForm` in the tests checks the export independently of both.
  Asymmetric or uint8 output, a quantized bias or a QOperator node is refused
  outright. The numeric test compares on calibration inputs (a held-out input
  is clipped, which is calibration's business) and carries a control that
  doubled scales fail the tolerance.
- **The form is what TensorRT sees, after ONNX Runtime.** Every session runs
  ONNX Runtime's level-1 transformers before TensorRT gets its nodes, and
  `WeightBiasQuantization` turns the 1-D bias of a quantized Conv,
  ConvTranspose or Gemm (or a MatMul it fuses with its Add) whose output goes
  straight into a QuantizeLinear into an int32 DequantizeLinear, which
  TensorRT does not run. `quantize/bias.go` mirrors its conditions
  (`quantizedBiases`, from `weight_bias_quantization.cc` and
  `matmul_add_fusion.cc` of the supported ONNX Runtime), `detachBiases` keeps
  such biases out of its reach, and `CheckTensorRT` refuses a model where one
  is left. `bias_test.go` saves ONNX Runtime's level-1 graph and checks it has
  no int32 DequantizeLinear, with the v0.3.0 form as the control that does.
  When ONNX Runtime moves, read those two files again.
- **The opset is raised only when safe.** `opsetChanges` in
  `quantize/check.go` lists the operators whose meaning changed between 10 and
  13; adding a raise path means adding the operators that change on it, from
  the ONNX operator changelog, not from memory.
- **`rewrite.UpgradeOpset` enumerates, it does not guess.** `opsetConverted`
  (`rewrite/opset.go`) is each operator rewritten and the opset its signature
  changed at; `opsetUnchanged` is each operator that computes the same from 7
  to 13. An operator on neither list is refused. Adding one means reading its
  sections in `docs/Changelog.md` of the ONNX release for every version
  between 7 and the target, diffing the attributes, inputs and description,
  and adding a case to `opsetCases` (`tests/Feature/opset_test.go`) that ORT
  runs at the old opset and the new. Raising the target past 13 is the same
  work for every operator, Reduce* (axes at 18) and Pad (axes at 18) first.
  Rank and element type come from `infer` (`rewrite/infer.go`), which answers
  unknown rather than guess; a rewrite that needs one takes the form that
  does not, or refuses.
- **Weights are constants only when the graph says so.** An initializer that
  is also a graph input can be overridden, so `graph.Graph.Constant` and
  `quantize.QDQ` do not treat it as a weight; `rewrite.InitializersOutOfInputs`
  is the explicit step that makes it one. A `Constant` node's value is a
  weight. `DropoutToIdentity` refuses a Dropout that could drop.
- **Real models are checked outside the repository.** `real_models_test.go`
  rewrites the exports that `ARANDU_ONNX_OPSET_UPGRADE_MODEL` and
  `ARANDU_ONNX_WEIGHT_INPUTS_MODEL` (with their `_SAMPLES` directories) name,
  and skips when they are unset. This repository is public: never commit a
  model, a sample, a table, an export, a digest of one, a dataset's name or a
  path of the machine that ran it, in code, docs or commit messages.
- **The commands open models only through manifests.** Source digest checked
  before decoding, derived model opened through a manifest with its own digest
  for the CPU comparison, nothing written when a check fails, and the printed
  manifest is what the file is shipped with. `quantization: qdq` in a manifest
  makes `Open` run `CheckTensorRT`; an int8 manifest without it is unchecked,
  as in v0.2.0.
