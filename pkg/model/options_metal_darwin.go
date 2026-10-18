//go:build darwin && metal

package model

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/device/metal"
)

// openMetalGPU opens the Metal backend for NGL>0 on macOS (build -tags metal).
// While Metal kernels are scaffold (Backend.Ready() == false) the device is closed and offload is rejected with a clear error; once kernels land and Ready() turns true, -ngl works with no further wiring.
func openMetalGPU() (device.Backend, error) {
	b, err := device.OpenMetal()
	if err != nil {
		return nil, fmt.Errorf("model: GPU offload (NGL>0) on macOS: %w", err)
	}

	if r, ok := b.(interface{ Ready() bool }); ok && !r.Ready() {
		_ = b.Close()

		return nil, fmt.Errorf("model: GPU offload (NGL>0) on macOS: Metal kernels not implemented yet (%w); use NGL=0 for CPU", metal.ErrUnavailable)
	}

	return b, nil
}
