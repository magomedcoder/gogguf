//go:build darwin && metal

package model

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/device/metal"
)

// openMetalGPU opens the Metal backend for NGL>0 on macOS (build -tags metal, cgo).
// With the B1 MSL runtime (Backend.Ready() == true) -ngl runs the decode matvec/RMSNorm/RoPE/SwiGLU/attention/KV ops on the GPU; if the runtime is not built in (cgo disabled) Ready() is false, the device is closed and offload is rejected with a clear error.
func openMetalGPU() (device.Backend, error) {
	b, err := device.OpenMetal()
	if err != nil {
		return nil, fmt.Errorf("model: GPU offload (NGL>0) on macOS: %w", err)
	}

	if r, ok := b.(interface{ Ready() bool }); ok && !r.Ready() {
		_ = b.Close()

		return nil, fmt.Errorf("model: GPU offload (NGL>0) on macOS: Metal kernels not built in (needs cgo + Xcode command line tools) (%w); use NGL=0 for CPU", metal.ErrUnavailable)
	}

	return b, nil
}
