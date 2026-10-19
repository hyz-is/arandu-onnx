//go:build linux

package onnx

import (
	"os"
	"syscall"
)

// The primitives capturePlacement needs. Linux has dup3 on every
// architecture and dup2 only on some, so dup3 is what is used.

func dupStderr() (int, error) { return syscall.Dup(int(os.Stderr.Fd())) }

func redirectStderr(w *os.File) error { return syscall.Dup3(int(w.Fd()), int(os.Stderr.Fd()), 0) }

func restoreStderr(saved int) error { return syscall.Dup3(saved, int(os.Stderr.Fd()), 0) }

func closeFD(fd int) { _ = syscall.Close(fd) }
