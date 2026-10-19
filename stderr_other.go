//go:build !linux && !darwin && !dragonfly && !freebsd && !netbsd && !openbsd

package onnx

import (
	"errors"
	"os"
)

// Without a way to duplicate file descriptors, what the C library writes to
// standard error cannot be captured, and a placement report comes back not
// captured. RequireAccelerated refuses that rather than reading it as clean.

var errNoCapture = errors.New("onnx: standard error cannot be captured on this platform")

func dupStderr() (int, error) { return 0, errNoCapture }

func redirectStderr(*os.File) error { return errNoCapture }

func restoreStderr(int) error { return errNoCapture }

func closeFD(int) {}
