package gpu

import (
	"errors"
	"strings"
)

var (
	// ErrUnavailable CUDA не собран или драйвер недоступен
	ErrUnavailable = errors.New("gpu: CUDA недоступна")
	// ErrInvalidNGL некорректное число слоёв для offload
	ErrInvalidNGL = errors.New("gpu: ngl должен быть >= 0")
	// ErrInvalidDevice некорректный ordinal устройства (-dev)
	ErrInvalidDevice = errors.New("gpu: номер устройства должен быть >= 0")
	// ErrMetalUnavailable Metal-backend не собран или доступен только как заглушка
	ErrMetalUnavailable = errors.New("gpu: Metal недоступен (нужны macOS и сборка -tags metal)")
	// ErrOutOfMemory не хватило VRAM при upload весов / буферов на устройство
	ErrOutOfMemory = errors.New("gpu: не хватило VRAM")
)

// IsOutOfMemory сообщает, что ошибка похожа на исчерпание VRAM при upload
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

	// CUDA Driver: cuMemAlloc при upload historically возвращает код -1
	return strings.Contains(msg, "upload") && strings.Contains(msg, "код -1")
}
