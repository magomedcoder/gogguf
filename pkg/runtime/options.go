package runtime

import "github.com/magomedcoder/gogguf/pkg/model"

// Options задаёт параметры загрузки inference-движка
type Options struct {
	NGL         int       // число transformer-слоёв на GPU (флаг -ngl)
	GPUMaxSeq   int       // макс. длина GPU KV-cache (0 = авто, до 4096)
	NBatch      int       // размер chunk prefill (флаг -b / --n-batch; 0/1 = по токену)
	GPUDevice   int       // ordinal CUDA-устройства (флаг -dev N)
	GPUDevices  []int     // устройства для layer split (флаг -dev 0,1)
	TensorSplit []float64 // пропорции слоёв по устройствам (флаг -tensor-split)
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
