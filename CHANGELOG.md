# Changelog

Everything worth knowing about a release of Arandu ONNX is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

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
