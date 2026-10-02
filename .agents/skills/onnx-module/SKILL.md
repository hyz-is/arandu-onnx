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

A provider is a name in `config.go`, a case in `appendProvider`, and a line in
the README table. It must fail soft: `probeProviders` records why it is
unavailable and `createSession` falls back to the CPU. A provider that can stop
the application from starting is refused.

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
