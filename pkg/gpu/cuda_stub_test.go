//go:build !cuda

package gpu

import (
	"errors"
	"testing"
)

// Без тега cuda OpenCUDA всегда возвращает ErrUnavailable
func TestOpenCUDAStub(t *testing.T) {
	_, err := OpenCUDA()
	if err != ErrUnavailable {
		t.Fatalf("OpenCUDA() = %v, ожидали ErrUnavailable", err)
	}
}

// Выбор устройства и multi-GPU без тега cuda тоже недоступны, но некорректный ordinal отсекается до драйвера
func TestOpenCUDADeviceStub(t *testing.T) {
	if _, err := OpenCUDADevice(1); err != ErrUnavailable {
		t.Fatalf("OpenCUDADevice(1) = %v, ожидали ErrUnavailable", err)
	}

	if _, err := OpenCUDADevice(-1); !errors.Is(err, ErrInvalidDevice) {
		t.Fatalf("OpenCUDADevice(-1) = %v, ожидали ErrInvalidDevice", err)
	}

	if _, err := OpenCUDADevices([]int{0, 1}); err != ErrUnavailable {
		t.Fatalf("OpenCUDADevices([0 1]) = %v, ожидали ErrUnavailable", err)
	}

	if _, err := OpenCUDADevicesSplit([]int{0, -2}, nil); !errors.Is(err, ErrInvalidDevice) {
		t.Fatalf("OpenCUDADevicesSplit([0 -2]) = %v, ожидали ErrInvalidDevice", err)
	}
}
