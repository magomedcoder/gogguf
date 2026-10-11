//go:build windows

package main

import "os"

func extraShutdownSignals() []os.Signal {
	return nil
}
