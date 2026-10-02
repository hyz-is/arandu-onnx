# Skills

Procedures an assistant follows when working on this package.

They live in `.agents/skills/<name>/SKILL.md`, which is the path the coding
assistants read from — Cursor, Codex, Cline, Copilot, Gemini CLI, Amp, OpenCode,
Warp, Zed and the rest all look there. It is one directory rather than a file
per vendor, so a skill written once is read by whatever this package is being
written with.

Each file opens with frontmatter carrying a `name` and a `description`. The
`name` has to equal the directory name exactly, or the skill is not loaded. The
description is what a tool reads to decide whether the skill is relevant, so it
names the situation you are in rather than the topic it covers.

| skill | when it fires |
| --- | --- |
| `onnx-module` | changing the runtime, manifests, sessions, tensors or image helpers |
| `onnx-release` | the gates, the manifest, a dependency, a version, a tag |
| `onnx-vault-notes` | writing the note, the gap or the journal entry, when this checkout sits inside the Arandu Obsidian vault |
| `onnx-package` | installing and wiring this package **into an application** |

The last one has a different audience from the other three, and that is on
purpose: it travels with the package so that an assistant working in somebody
else's project — the one running `go get` — has the build requirement, the
wiring and the manifest format in front of it instead of guessing.

`onnx-vault-notes` fires on a condition rather than on a task: it applies
only when `MOC-arandu.md`, `plans/cmd/audit-vault/` and `45-modules/` are actually
beside this checkout. Outside the vault it is inert, and it says so first, so a
package cloned somewhere else never grows a folder tree imitating one.

## Why these exist

The audience of the first three is somebody changing the package. The common
failure modes are a threshold that turns a tensor into a decision, a model
loaded by path without its manifest, a Python script to produce a fixture, and
an environment variable for the library path. None belongs here; the first two
are how an engine's evidence stops being evidence.

## Adding your own

A skill in this directory is yours and travels with the repository. Keep it a
procedure rather than a description: a file that says "read the documentation"
never changes what anybody does.
