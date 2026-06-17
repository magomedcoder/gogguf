package gpu

import "errors"

var (
	// ErrUnavailable CUDA не собран или драйвер недоступен
	ErrUnavailable = errors.New("gpu: CUDA недоступна")
	// ErrInvalidNGL некорректное число слоёв для offload
	ErrInvalidNGL = errors.New("gpu: ngl должен быть >= 0")
)
