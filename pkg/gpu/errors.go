package gpu

import (
	"errors"
	"strings"
)

var (
	// ErrUnavailable CUDA not built or driver unavailable
	ErrUnavailable = errors.New("gpu: CUDA недоступна")
	// ErrInvalidNGL invalid layer count for offload
	ErrInvalidNGL = errors.New("gpu: ngl должен быть >= 0")
	// ErrInvalidDevice invalid device ordinal (-dev)
	ErrInvalidDevice = errors.New("gpu: номер устройства должен быть >= 0")
	// ErrMetalUnavailable Metal backend not built or only available as stub
	ErrMetalUnavailable = errors.New("gpu: Metal недоступен (нужны macOS и сборка -tags metal)")
	// ErrOutOfMemory not enough VRAM when uploading weights/buffers to device
	ErrOutOfMemory = errors.New("gpu: не хватило VRAM")
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
	if strings.Contains(msg, "не хватило VRAM") {
		return true
	}

	// CUDA Driver: cuMemAlloc on upload historically returns code -1
	return strings.Contains(msg, "upload") && strings.Contains(msg, "код -1")
}
