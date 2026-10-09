package runtime

import "github.com/magomedcoder/gogguf/pkg/model"

// Options configures inference engine load parameters.
type Options struct {
	NGL         int       // number of transformer layers on GPU (flag -ngl)
	GPUMaxSeq   int       // max GPU KV-cache length (0 = auto, up to 4096)
	NBatch      int       // prefill chunk size (flag -b / --n-batch; 0/1 = one token at a time)
	GPUDevice   int       // CUDA device ordinal (flag -dev N)
	GPUDevices  []int     // devices for layer split (flag -dev 0,1)
	TensorSplit []float64 // layer proportions per device (flag -tensor-split)
}

func (o Options) modelOpts() model.Options {
	return model.Options{
		NGL:         o.NGL,
		GPUMaxSeq:   o.GPUMaxSeq,
		NBatch:      o.NBatch,
		GPUDevice:   o.GPUDevice,
		GPUDevices:  o.GPUDevices,
		TensorSplit: o.TensorSplit,
	}
}
