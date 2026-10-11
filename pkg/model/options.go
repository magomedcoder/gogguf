package model

import (
	"fmt"
	"runtime"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/exec/gpuresid"
)

// Options sets model load parameters (library-oriented; CLI maps flags into these fields).
type Options struct {
	NGL         int            // transformer layers on GPU (0 = CPU)
	GPUMaxSeq   int            // max KV length on GPU (0 = min(context, 4096))
	NBatch      int            // prefill chunk size; effective for Qwen3 dense only (others ignore)
	GPUDevice   int            // CUDA device ordinal when GPUDevices is empty
	GPUDevices  []int          // devices for layer split; nil = single GPUDevice
	TensorSplit []float64      // layer proportions per device; nil = even split
	GPU         device.Backend // if nil and NGL > 0, CUDA is opened

	// Residency toggles fused GPU paths. nil = enabled.
	HiddenResidency  *bool
	QKVResidency     *bool
	AttnFFNResidency *bool
}

// Devices returns the device list for offload (always non-empty when NGL > 0)
func (o Options) Devices() []int {
	if len(o.GPUDevices) > 0 {
		return o.GPUDevices
	}

	return []int{o.GPUDevice}
}

// Normalize validates options, applies residency overrides, and opens CUDA when needed.
func (o *Options) Normalize() error {
	gpuresid.Configure(o.HiddenResidency, o.QKVResidency, o.AttnFFNResidency)

	if o.NGL < 0 {
		return device.ErrInvalidNGL
	}

	if o.GPUDevice < 0 {
		return device.ErrInvalidDevice
	}

	if o.NGL == 0 {
		o.GPU = nil
		return nil
	}

	if o.GPU != nil {
		return nil
	}

	if runtime.GOOS == "darwin" {
		return fmt.Errorf("model: GPU offload (NGL>0) on macOS requires Metal kernels (not implemented); use NGL=0 for CPU")
	}

	g, err := device.OpenCUDADevicesSplit(o.Devices(), o.TensorSplit)
	if err != nil {
		return err
	}

	o.GPU = g

	return nil
}
