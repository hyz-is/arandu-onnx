# Security policy

## Supported versions

The latest minor release of Arandu ONNX receives security fixes. Older minor
releases do not: a published Go module version is immutable, so a fix is a new
release rather than a moved tag.

## Reporting a vulnerability

Report it privately, through GitHub's advisory form:

<https://github.com/kuaa-io/arandu-onnx/security/advisories/new>

Do not open a public issue and do not describe the problem in a pull request.
A report that arrives in public is a report every reader of this repository can
act on before there is a release to upgrade to.

Include what you need to reproduce it: the version, the configuration, and the
source that triggers it.

Expect an acknowledgement within a few days. If the report is confirmed, the
fix, the release and the advisory are published together, and you are credited
unless you ask not to be.

## What is in scope

Anything that lets a model run that its manifest did not admit, or lets input
reach further than a tensor should. In particular:

- a way to open a session on a file whose SHA-256 is not the manifest's, or on
  a manifest whose status is `blocked`;
- a manifest `file` that reads outside the model directory;
- an input tensor or an image that crashes the process, reads out of bounds, or
  costs memory or time out of proportion to its size;
- a race that lets a session run after it was closed.

## What is not

- A vulnerability in ONNX Runtime itself. Report it to that project; a release
  here that requires a fixed library version follows.
- What an application decides from the outputs. This package returns tensors.
