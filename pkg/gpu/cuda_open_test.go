//go:build cuda

package gpu

import (
	"errors"
	"testing"
)

// With cuda tag OpenCUDA returns backend (GPU present) or ErrUnavailable
func TestOpenCUDATagged(t *testing.T) {
	b, err := OpenCUDA()
	if err != nil {
		if err != ErrUnavailable {
			t.Logf("OpenCUDA(): %v", err)
		}

		t.Skip("CUDA недоступна")
	}

	defer b.Close()

	if b == nil {
		t.Fatal("OpenCUDA() вернул nil backend без ошибки")
	}
}

// -dev N: device 0 opens; an out-of-range ordinal is rejected
func TestOpenCUDADeviceTagged(t *testing.T) {
	b, err := OpenCUDADevice(0)
	if err != nil {
		t.Skipf("CUDA недоступна: %v", err)
	}
	defer b.Close()

	used, total, err := b.VRAMInfo()
	if err != nil {
		t.Fatalf("VRAMInfo: %v", err)
	}

	if total == 0 || used > total {
		t.Fatalf("VRAMInfo = %d/%d байт", used, total)
	}

	t.Logf("%s: VRAM %d/%d MB", b.Name(), used>>20, total>>20)

	if _, err := OpenCUDADevice(1024); err == nil {
		t.Fatal("OpenCUDADevice(1024): ожидали ошибку диапазона")
	}

	if _, err := OpenCUDADevice(-1); !errors.Is(err, ErrInvalidDevice) {
		t.Fatalf("OpenCUDADevice(-1) = %v, ожидали ErrInvalidDevice", err)
	}
}

// Single-device list gives normal single-GPU backend (no multi wrapper)
func TestOpenCUDADevicesSingle(t *testing.T) {
	b, err := OpenCUDADevices([]int{0})
	if err != nil {
		t.Skipf("CUDA недоступна: %v", err)
	}
	defer b.Close()

	if _, ok := b.(*MultiBackend); ok {
		t.Fatal("OpenCUDADevices([0]) вернул MultiBackend вместо одного устройства")
	}

	if Describe(b) != b.Name() {
		t.Fatalf("Describe = %q, Name = %q", Describe(b), b.Name())
	}
}
