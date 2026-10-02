---
name: onnx-release
description: Get this Arandu package green, declared and published. Use when the request is to "run the gates", "check it builds", "the tests fail", "go build says the directory prefix does not contain modules listed in go.work", "aru doctor says this is not an Arandu project", "update arandu.mod.toml", "add a dependency", "bump the framework", "tag a release", "publish it", "go get cannot find the module", "write the changelog", or before opening a pull request. Covers the four gates and why each filter is there, CGO_ENABLED and the ONNX Runtime library,, what the manifest declares and who checks it, the immutability of a published module version, and what breaks somebody else's build.
license: MIT
---

# Getting it green and out

## The four gates

Nothing is finished until all four exit zero.

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

`GOWORK=off` first, and here it is not a habit. This checkout may sit beside a
Go workspace that lists the framework repositories and does not list this one.
When it does, `go build ./...` exits 1 before compiling anything:

```
pattern ./...: directory prefix . does not contain modules listed in go.work
or their selected dependencies
```

That is the workspace refusing to answer for a module it was not told about, and
it says nothing about the code. With the workspace off, the module resolves the
versions in `go.mod` — which is what CI compiles against, and what somebody's
`go get` will get. A pass obtained with a workspace active is a pass against
whatever happens to be checked out beside you.

`aru doctor` is not a gate. It exits 1 here with *"this is not an Arandu
project: no go.mod, main.go and arandu.toml together"* — it reads applications,
and this is a library.

`CGO_ENABLED=1` is not optional: the ONNX Runtime binding is cgo. The tests
that load a model need the shared library at `DefaultSharedLibraryPath()`, and
skip with the loader's message where it is absent. A run where they skipped
has not tested the runtime; read the skip lines before calling it green.

CI runs the same four.

## What the manifest declares, and who reads it

`arandu.mod.toml` is a promise, and the check that gives it weight runs here.
Nowhere else does: `aru doctor` audits the tree of the application it is run
inside and never loads a dependency, so the manifest of an installed package is
opened by nothing but that package's own suite.

```toml
name = "kuaa-io/onnx"
framework = ">= 0.50"
profiles = ["conventional", "performance"]

[permissions]
network = false
filesystem = true
exec = false
migrations = false
```

`TestTheDeclaredCapabilitiesAreWhatTheCodeDoes`, in `tests/Unit/audit_test.go`,
compares the four permissions against what the code *calls*, by AST —
`os.WriteFile`, `os.ReadFile`, `filepath.WalkDir`, `exec.Command`, `http.Get`,
`.Do`, `net.Dial`, and a method named `Migrations` on a receiver. It looks at
calls rather than imports because `net/http` is imported by everything that has
a route and says nothing about whether it calls out. A file behind a build tag
is left out, because the compiler never puts it in the module.

Two findings come out of that comparison. The names and severities below are
`aru doctor`'s, from the day an application runs the same comparison over its
own code; here both fail the suite, because `go test` has one outcome:

| finding | severity | what it means |
| --- | --- | --- |
| `permission-not-declared` | error | the code does it and the manifest says `false`. Whoever installed this agreed to a module that does not |
| `permission-not-used` | warning | the manifest says `true` and nothing does it. Asking for more than you need is how a permission model erodes into everyone declaring everything |

So a dependency or a call added here is a line changed in `arandu.mod.toml` in
the same commit. `filesystem = true` is the one this package declares: it reads
model files and checks that the library path exists. The cgo link to ONNX
Runtime has no key in the manifest, so its comment says it.

The `framework` line is a floor, and it is a second declaration of something
`go.mod` already says. Move both together when you bump.

## Adding a dependency

Do not, without an argument. This module goes into other people's builds: a
second `require` is a download and an audit surface for every project that
installs the package, and the framework's own rule is a core with no third-party
dependency beyond `golang.org/x/crypto`.

What is already here is the framework, the components module it is built on,
and `github.com/yalue/onnxruntime_go` -- the cgo binding to ONNX Runtime. Its
version is pinned to the release whose embedded header requests a C API version
the supported library serves (v1.31.0 requests 26; ONNX Runtime 1.26 and later
serve it). Moving it moves the minimum library version, which is a changelog
and upgrade-note line:

```sh
export GOWORK=off
go list -m -f '{{if and (not .Indirect) (not .Main)}}{{.Path}} {{.Version}}{{end}}' all
```

If something genuinely has to come in, it goes with a note in the pull request
saying what it does that the framework does not, and the manifest changes in the
same commit if it can reach the network, the filesystem or another process.

## Publishing

A published Go module version is **immutable**. The proxy serves it forever, and
a mistake is corrected by another release, never by moving a tag. That is the
reason `CHANGELOG.md` says so and the reason a manifest key is fixed once it
has shipped.

Before tagging:

- the four gates pass with `GOWORK=off` and `CGO_ENABLED=1`, with the runtime
  tests run rather than skipped;
- `arandu.mod.toml` matches the code;
- `CHANGELOG.md` has the entry, under the version, in the Keep a Changelog
  sections — `Added`, `Changed`, `Fixed`, `Removed`.

While the version starts with `v0.`, breaking is allowed — freezing a shape this
early is worse. Breaking *quietly* is not. A field that stops existing is a
build that stops in somebody else's repository, weeks later, when they upgrade,
with an error about a struct literal they did not write. Say what it was, what
it is, and what they have to write instead.

Run `apidiff` against the release baseline and enumerate every removal.

Three changes break an installer without touching a signature, and each one is
worth a line in the changelog:

- **A manifest field.** Manifests are JSON files beside models in other
  repositories, and unknown keys are refused, so a renamed key is every
  installer's models refusing to open.
- **The minimum library version.** Moving the binding can make a machine that
  worked fail `Init`.
- **A default.** `DefaultSharedLibraryPath()` and the provider order are what an
  application gets by leaving the field empty, so moving one moves behaviour in
  every project that never mentioned it.

## Reporting a vulnerability

Not in an issue and not in a pull request. `SECURITY.md` has the private
advisory address. Anything that lets a model run without its digest verified,
a blocked model load, a file outside the model directory be read, or an input
crash the process is in scope.
