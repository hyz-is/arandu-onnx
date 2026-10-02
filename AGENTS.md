# Working on Arandu ONNX

This is an Arandu package that runs ONNX models through ONNX Runtime and
answers with tensors. It is the shared runtime KUAA's engine modules
(`arandu-paddleocr`, `arandu-face`, `arandu-liveness`) build on: a runtime, a
model registry with provenance, sessions, and the image helpers vision engines
need.
It is a Go module somebody `go get`s and registers by hand in their own
`bootstrap/app.go`, which is the whole difference from working in an
application. There is no service provider, no container and no discovery — if a
line of wiring is not written in the installer's repository, it does not happen.

Read `.agents/skills/` before writing code. Each skill is a procedure, and the
one you need is named by the situation you are in.

## The KUAA rules

These come from the project, not from taste. A change that breaks one is not
merged.

1. **No decisions in an engine, and none here.** This package returns
   technical tensors and the `ModelInfo` that produced them. It never returns a
   verdict, a threshold applied, an "approved" or a "match". Thresholds and
   decisions belong to the application's decision layer, which records the
   evidence this package attaches.
2. **No Python.** Not as a runtime, not in a gate, not in a test, not to
   generate a fixture. Models arrive as already-exported ONNX files; the tests
   write their models as hand-encoded protobuf in Go.
3. **Model provenance is mandatory.** A model is opened only through a
   `Manifest` that records its digest, source, licences, commercial use,
   redistribution, modification, status and precision. `Open` verifies the
   SHA-256 before ONNX Runtime sees the bytes and refuses `blocked`. There is no
   path that loads a file without a manifest, and none may be added.
4. **Configuration is typed.** `Config` comes from the application. This
   package reads no environment variable.

## The gates

Nothing is finished until all four exit zero.

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

`CGO_ENABLED=1` is required: the binding is cgo. The tests that load models need
the ONNX Runtime shared library at `DefaultSharedLibraryPath()`; where it is
absent they skip with the loader's own message, and the skip is a reason to fix
the machine, not a green result for the runtime.

`GOWORK=off` matters because this checkout may sit beside a Go workspace that
does not list it; with the workspace on, every command above fails before it
compiles anything.

`aru doctor` reads applications, not libraries. Run here it exits 1 with
"this is not an Arandu project", and it does not read an installed package
either. That is why `tests/Unit/audit_test.go` compares `arandu.mod.toml` with
what the code calls.

## What this repository holds

```
module.go    registration: Name, Routes (none), Boot, Health, Close, Runtime
config.go    what the application passes in, and the default library path
runtime.go   the library, the providers, Open and the contract check
manifest.go  the provenance and contract of a model file
session.go   Run, ModelInfo and Result
tensor.go    the tensor exchanged with a model
image.go     decode, resize, letterbox, crop, NCHW
exif.go      the EXIF orientation of a JPEG
```

Tests live under `tests/Unit` (manifests, tensors, images, the audit) and
`tests/Feature` (the module, and every test that loads a model).

## What does not exist here, on purpose

| A model reaches for | What is here instead |
| --- | --- |
| a threshold, a score cut-off, a boolean "is live" | the raw output tensor. The decision layer applies thresholds |
| a Python export or conversion script | an ONNX file and its manifest, obtained elsewhere |
| loading a model by path alone | `Runtime.Open(manifest, dir)` |
| an environment variable for the library path | `Config.SharedLibraryPath` |
| a route that runs a model | nothing. Engines are reached through the application's controllers |
| a table of models | the manifest beside the file. The registry is the files and their manifests |
| a second ONNX binding or a hand-written cgo layer | `github.com/yalue/onnxruntime_go`, pinned to the release whose C API version the supported ONNX Runtime serves |

## The binding and the library

`onnxruntime_go` embeds the ONNX Runtime C header and asks the loaded library
for `ORT_API_VERSION` from it. A library older than that version returns no
API and `Init` fails with `ErrLibraryUnavailable`. v1.31.0 requests 26, which
ONNX Runtime 1.26 and later serve; v1.32.0 and later request 28 or more. Moving
the binding is moving the minimum library version: say so in `CHANGELOG.md`
and `UPGRADE.md`.

The binding keeps the environment in package globals, so one process has one
library. `runtime.go` shares it between `Runtime` values by reference count and
refuses a second path.

## Writing code

Everything in the source is in English: identifiers, doc comments, internal
comments, error messages, log messages, and the names and messages of tests.

Every exported symbol carries a doc comment, and the comment documents the
symbol and nothing else. A date, an issue number, a version in progress or the
name of another repository does not belong there.

Tests go under `tests/`, in a capitalized category directory declaring a
lowercase external package: `tests/Unit` holds `package unit_test`,
`tests/Feature` holds `package feature_test`. Both import the package by its
module path, which is what makes them see exactly what a caller sees.
