package model

import "github.com/magomedcoder/gguf.go/pkg/gpu"

// Options задаёт параметры загрузки модели
type Options struct {
	NGL int         // число transformer-слоёв для offload на GPU (-ngl)
	GPU gpu.Backend // backend; если nil и NGL > 0, будет попытка OpenCUDA()
}

// Normalize проверяет опции и при необходимости открывает CUDA
func (o *Options) Normalize() error {
	if o.NGL < 0 {
		return gpu.ErrInvalidNGL
	}

	if o.NGL == 0 {
		o.GPU = nil
		return nil
	}

	if o.GPU != nil {
		return nil
	}

	g, err := gpu.OpenCUDA()
	if err != nil {
		return err
	}

	o.GPU = g

	return nil
}
