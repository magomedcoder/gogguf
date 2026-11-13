//go:build !(darwin && metal)

package model

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
)

// openMetalGPU without darwin+metal build: GPU offload on macOS is rejected (NGL=0 runs on CPU; build with -tags metal for the Metal B1 decode path)
func openMetalGPU() (device.Backend, error) {
	return nil, fmt.Errorf("model: GPU offload (NGL>0) on macOS requires a build with -tags metal (macOS, cgo; Metal B1 decode path); use NGL=0 for CPU")
}
