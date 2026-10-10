package gpu

import (
	"errors"
	"strings"
)

var (
	// ErrUnavailable CUDA not built or driver unavailable
	ErrUnavailable = errors.New("gpu: CUDA unavailable")
	// ErrInvalidNGL invalid layer count for offload
	ErrInvalidNGL = errors.New("gpu: ngl must be >= 0")
	// ErrInvalidDevice invalid device ordinal (-dev)
	ErrInvalidDevice = errors.New("gpu: device index must be >= 0")
	// ErrMetalUnavailable Metal backend not built or only available as stub
	ErrMetalUnavailable = errors.New("gpu: Metal unavailable (requires macOS and build -tags metal)")
	// ErrOutOfMemory not enough VRAM when uploading weights/buffers to device
	ErrOutOfMemory = errors.New("gpu: out of VRAM")
)

// IsOutOfMemory reports whether the error looks like VRAM exhaustion on upload
func IsOutOfMemory(err error) bool {
	if err == nil {
		return false
	}

	if errors.Is(err, ErrOutOfMemory) {
		return true
	}

	msg := err.Error()
	if strings.Contains(msg, "out of VRAM") {
		return true
	}

	// CUDA Driver: cuMemAlloc on upload historically returns code -1
	return strings.Contains(msg, "upload") && strings.Contains(msg, "code -1")
}
