//go:build cuda

package gpu

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/gpu/cuda"
)

// OpenCUDA инициализирует CUDA через Driver API (libcuda) на устройстве 0
func OpenCUDA() (Backend, error) {
	return cuda.Open()
}

// OpenCUDADevice инициализирует CUDA на устройстве с ordinal device (флаг -dev).
// Нумерация - после фильтра CUDA_VISIBLE_DEVICES
func OpenCUDADevice(device int) (Backend, error) {
	if device < 0 {
		return nil, fmt.Errorf("%w: %d", ErrInvalidDevice, device)
	}

	return cuda.OpenDevice(device)
}

// OpenCUDADevices открывает несколько устройств и распределяет слои между ними.
// Один элемент в списке = обычный однокарточный backend без обёртки
func OpenCUDADevices(devices []int) (Backend, error) {
	return OpenCUDADevicesSplit(devices, nil)
}

// OpenCUDADevicesSplit как OpenCUDADevices, но с пропорциями слоёв (-tensor-split)
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
