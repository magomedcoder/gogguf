//go:build cuda

package gpu

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/gpu/cuda"
)

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
