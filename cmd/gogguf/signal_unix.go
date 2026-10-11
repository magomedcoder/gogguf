//go:build unix

package main

import (
	"os"
	"syscall"
)

func extraShutdownSignals() []os.Signal {
	return []os.Signal{syscall.SIGTERM}
}
