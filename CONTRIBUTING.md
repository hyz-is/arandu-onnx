# Contributing

Thank you for considering a contribution.

## Before you open a pull request

Run the four gates. They are what CI runs, and a pull request that fails one of
them fails for everybody who pulls after it.

```bash
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

The two filters on `gofmt` are not preference. `gofmt` is the only tool in the
chain that ignores build tags, so it reads files the compiler deliberately
excludes; `testdata/` holds fixtures that are invalid on purpose.

The tests that load a model need the ONNX Runtime shared library and skip,
saying why, where it is absent. A pull request that touches the runtime is run
with the library present.

## What a change has to keep true

Four properties are the reason this package is what it is. A change that breaks
one of them will not be merged, whatever else it improves.

1. **Tensors, never decisions.** No threshold, verdict or score cut-off. The
   application's decision layer applies them and records `ModelInfo` beside.
2. **No model without its manifest.** `Runtime.Open` refuses a blocked model
   before reading it and checks the SHA-256 before ONNX Runtime sees the bytes.
   No path around either is added.
3. **No Python.** Not at runtime, not in a gate, not to generate a fixture.
4. **`arandu.mod.toml` matches the code.** Adding an outbound call, a file
   write or a process means declaring it there in the same commit.

## Style

Everything in the source is in English: identifiers, doc comments, internal
comments, error messages, log messages, and the names and messages of tests.
`pkg.go.dev` publishes the doc comments, and its readers are users of this
package.

Every exported symbol carries a doc comment saying what it does. The comment
documents the symbol and nothing else: no dates, no issue numbers, no
references to other repositories. Why a signature is what it is belongs there
when it is a fact about the code; anything else belongs in the pull request.

## Commits

One commit per logical change. Formatting is not mixed with behaviour, because
a diff that does both is a diff nobody reviews.

## Reporting a vulnerability

Do not open an issue. See [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence, the same as the rest of github.com/kuaa-io/arandu-onnx.
