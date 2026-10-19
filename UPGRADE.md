# Upgrade Guide

Every release that changes what an installer has to write is described here,
newest first. A release with nothing to change still has a heading, so this file
and [CHANGELOG.md](CHANGELOG.md) describe the same set of versions.

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
