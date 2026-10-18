//go:build cuda

package device

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device/cuda"
)

// optional capabilities of the CUDA backend (compile-time checks)
var (
	_ MLABackend              = (*cuda.Backend)(nil)
	_ KVBatchAttentionBackend = (*cuda.Backend)(nil)
	_ MLABackend              = (*MultiBackend)(nil)
	_ KVBatchAttentionBackend = (*MultiBackend)(nil)
)

func init() {
	defaultPeerHidden = cudaPeerHidden
}

// cudaPeerHidden moves resident hidden between two CUDA devices over P2P (cuMemcpyPeer); error = not moved (host fallback)
func cudaPeerHidden(src, dst Backend, n int) error {
	s, ok := src.(*cuda.Backend)
	if !ok {
		return fmt.Errorf("device: peer: %T is not a CUDA backend", src)
	}

	d, ok := dst.(*cuda.Backend)
	if !ok {
		return fmt.Errorf("device: peer: %T is not a CUDA backend", dst)
	}

	return s.PeerCopyResidTo(d, n)
}

// OpenCUDA initializes CUDA via Driver API (libcuda) on device 0
func OpenCUDA() (Backend, error) {
	return cuda.Open()
}

// OpenCUDADevice initializes CUDA on device with ordinal ( -dev flag).
// Numbering - after CUDA_VISIBLE_DEVICES filter
func OpenCUDADevice(device int) (Backend, error) {
	if device < 0 {
		return nil, fmt.Errorf("%w: %d", ErrInvalidDevice, device)
	}

	return cuda.OpenDevice(device)
}

// OpenCUDADevices opens multiple devices and splits layers between them.
// One list entry = normal single-GPU backend without wrapper
func OpenCUDADevices(devices []int) (Backend, error) {
	return OpenCUDADevicesSplit(devices, nil)
}

// OpenCUDADevicesSplit like OpenCUDADevices but with layer proportions (-tensor-split)
func OpenCUDADevicesSplit(devices []int, split []float64) (Backend, error) {
	devices, err := normalizeDeviceList(devices)
	if err != nil {
		return nil, err
	}

	if len(devices) == 1 {
		return cuda.OpenDevice(devices[0])
	}

	opened := make([]Backend, 0, len(devices))
	for _, dev := range devices {
		b, err := cuda.OpenDevice(dev)
		if err != nil {
			for _, o := range opened {
				_ = o.Close()
			}

			return nil, err
		}

		opened = append(opened, b)
	}

	multi, err := NewMultiBackend(opened, split)
	if err != nil {
		for _, o := range opened {
			_ = o.Close()
		}

		return nil, err
	}

	return multi, nil
}
