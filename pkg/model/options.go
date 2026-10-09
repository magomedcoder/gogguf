package model

import "github.com/magomedcoder/gogguf/pkg/gpu"

// Options sets model load parameters
type Options struct {
	NGL         int         // number of transformer layers to offload to GPU (-ngl)
	GPUMaxSeq   int         // max KV length on GPU (0 = min(context, 4096))
	NBatch      int         // chunk prefill size (0/1 = per token; Qwen3 CPU)
	GPUDevice   int         // CUDA device ordinal (-dev N); used when GPUDevices is empty
	GPUDevices  []int       // devices for layer split (-dev 0,1); nil = single GPUDevice
	TensorSplit []float64   // layer proportions per device (-tensor-split); nil = even split
	GPU         gpu.Backend // backend; if nil and NGL > 0, CUDA will be opened
}

// Devices returns the device list for offload (always non-empty when NGL > 0)
func (o Options) Devices() []int {
	if len(o.GPUDevices) > 0 {
		return o.GPUDevices
	}

	return []int{o.GPUDevice}
}

// Normalize validates options and opens CUDA when needed
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
