//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package onnx

import (
	"os"
	"syscall"
)

// The primitives capturePlacement needs, with the BSD system calls.

func dupStderr() (int, error) { return syscall.Dup(int(os.Stderr.Fd())) }

func redirectStderr(w *os.File) error { return syscall.Dup2(int(w.Fd()), int(os.Stderr.Fd())) }

func restoreStderr(saved int) error { return syscall.Dup2(saved, int(os.Stderr.Fd())) }

func closeFD(fd int) { _ = syscall.Close(fd) }
