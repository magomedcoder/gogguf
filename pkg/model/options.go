package model

import "github.com/magomedcoder/gogguf/pkg/gpu"

// Options задаёт параметры загрузки модели
type Options struct {
	NGL         int         // число transformer-слоёв для offload на GPU (-ngl)
	GPUMaxSeq   int         // макс. длина KV на GPU (0 = min(context, 4096))
	NBatch      int         // размер chunk prefill (0/1 = по токену; Qwen3 CPU)
	GPUDevice   int         // ordinal CUDA-устройства (-dev N); учитывается при пустом GPUDevices
	GPUDevices  []int       // устройства для layer split (-dev 0,1); nil = одно GPUDevice
	TensorSplit []float64   // пропорции слоёв по устройствам (-tensor-split); nil = равномерно
	GPU         gpu.Backend // backend; если nil и NGL > 0, будет попытка открыть CUDA
}

// Devices возвращает список устройств для offload (всегда непустой при NGL > 0)
func (o Options) Devices() []int {
	if len(o.GPUDevices) > 0 {
		return o.GPUDevices
	}

	return []int{o.GPUDevice}
}

// Normalize проверяет опции и при необходимости открывает CUDA
func (o *Options) Normalize() error {
	if o.NGL < 0 {
		return gpu.ErrInvalidNGL
	}

	if o.GPUDevice < 0 {
		return gpu.ErrInvalidDevice
	}

	if o.NGL == 0 {
		o.GPU = nil
		return nil
	}

	if o.GPU != nil {
		return nil
	}

	g, err := gpu.OpenCUDADevicesSplit(o.Devices(), o.TensorSplit)
	if err != nil {
		return err
	}

	o.GPU = g

	return nil
}
