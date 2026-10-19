---
name: onnx-module
description: Change this Arandu package's runtime, manifests, sessions or image helpers. Use when the request is to "support another element type", "add a provider", "add TensorRT", "add a manifest field", "load a model by path", "skip the digest check", "add a threshold", "return a decision", "add an image helper", "resize differently", "add a route", or when a change touches runtime.go, manifest.go, session.go, tensor.go, image.go, exif.go, config.go or module.go. Covers what may and may not change, and the tests that hold each property.
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
