# Upgrade Guide

Every release that changes what an installer has to write is described here,
newest first. A release with nothing to change still has a heading, so this file
and [CHANGELOG.md](CHANGELOG.md) describe the same set of versions.

## v0.4.0

Code and configuration need no change; QDQ exports made with v0.3.0 do:

- **Re-export every QDQ model v0.3.0 wrote**, with the same table and
  options. ONNX Runtime's `WeightBiasQuantization` rewrites the bias of a
  quantized Conv, ConvTranspose, Gemm or fused MatMul whose output goes
  straight into a QuantizeLinear into an int32 DequantizeLinear before
  TensorRT is handed the graph; TensorRT cannot run it, so those nodes run
  on the CPU and the TensorRT graph is split (placement shows
  `DequantizeLinear` on the CPU, named `<node>_bias_dq`). v0.4.0 writes those
  biases where the transformer does not take them, with the same scales
  (`ScalesSHA256` is unchanged). `CheckTensorRT` now refuses the old form,
  so a manifest declaring `"quantization": "qdq"` for such a file is refused
  at `Open` with `ErrContractMismatch` instead of being split on a GPU node:
  ship the new file and its new digest.

Three behaviours move, all toward doing more with the same input:

- `quantize.QDQ` keeps biases out of ONNX Runtime's bias quantization, as
  above; `Report.DetachedBiases` lists the ones it moved.
- `quantize.QDQ` now quantizes a node whose weight is a `Constant` node; in
  v0.3.0 it skipped it. An export of such a model has more nodes in INT8
  than before, and a new `ScalesSHA256`.
- Calibration no longer lists the outputs of `Constant` nodes and of nodes
  that read only constants. A table written by v0.3.0 keeps working; a new
  one has fewer entries, and its digest changes.

To export an old model (opset 7 to 12, weights listed as graph inputs, or a
Dropout) to QDQ:

1. `go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-rewrite@v0.4.0
   -manifest <manifest> -opset 13 -initializers-out-of-inputs
   -dropout-to-identity -out <model>.inference.onnx` (the flags the model
   needs), and check the difference it prints.
2. Calibrate the new file (`arandu-onnx-calibrate`) or, when only the opset
   or the inputs changed, reuse the old table with `-table`: the tensor names
   are the same.
3. `arandu-onnx-quantize -manifest <the printed manifest> ...` as in v0.3.0.

A manifest with the `derived_from` key needs v0.4.0: v0.3.0 refuses the
unknown key. Drop it to open the file with an older release.

## v0.3.0

Nothing has to change. Every addition is new: three packages (`graph`,
`rewrite`, `quantize`), two commands, and one optional manifest key.

To run a model with `Col2Im` entirely on TensorRT:

1. `go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-rewrite@v0.3.0
   -manifest <manifest> -out <model>.convtranspose.onnx`, and check the
   difference it prints.
2. Ship the new file with the manifest the command printed. It has its own
   SHA-256; a calibration table of the old file does not carry over, so
   calibrate the new one again if it is to run in INT8.

To move a model from a calibration table (implicit INT8, gone in TensorRT 11)
to QDQ:

1. `go run github.com/hyz-is/arandu-onnx/cmd/arandu-onnx-quantize@v0.3.0
   -manifest <float manifest> -out <model>.qdq.onnx`, with the manifest's
   `calibration` entry as the table and the exclusions the float table had.
2. Ship the new file with the printed manifest: `"precision": "int8"` and
   `"quantization": "qdq"`, no `calibration` entry. A manifest with the
   `quantization` key needs v0.3.0: v0.2.0 refuses the unknown key.
3. Measure it against FP16 on the card before switching, as with any INT8.

An int8 manifest without `quantization` opens exactly as in v0.2.0, unchecked.

## v0.2.0

Nothing has to change. A v0.1.0 configuration, manifest and call site builds
and behaves as before, with one difference on failure: when a session cannot
be built with every available accelerator, v0.2.0 tries the remaining ones
before the CPU alone.

To run on NVIDIA cards with TensorRT:

1. Install the GPU build of ONNX Runtime (1.26 or later) with its CUDA and
   TensorRT provider libraries, and the CUDA, cuDNN and TensorRT it was built
   with. TensorRT has to be below 11 for INT8 from a calibration table.
2. Name the providers `tensorrt`, `cuda`, `cpu` in `Config.Providers`, and set
   `Config.TensorRT` with an absolute `CacheDir` on a volume private to the
   application. `INT8`, `EngineCache` and `TimingCache` are refused without
   it.
3. For a float model that should run in INT8, write its table with
   `arandu-onnx-calibrate` or `Runtime.Calibrate`, put it beside the model, and
   add the printed `calibration` entry to its manifest. A manifest with a
   `calibration` entry needs v0.2.0: v0.1.0 refuses the unknown key.
4. Open one session per card with `OpenOn(manifest, dir, device)`, and set
   `Placement.RequireAccelerated` where a CPU fallback must stop the model
   from serving.

Code that compares a whole `ModelInfo` value with a literal keeps compiling;
the two new fields, `Device` and `Calibration`, are zero for CPU and Core ML
sessions.

## v0.1.0

The first public release. Nothing to upgrade from.

A checkout that imported the module before it was published under its current
path replaces the old import path with `github.com/hyz-is/arandu-onnx` and
requires `v0.1.0`.
