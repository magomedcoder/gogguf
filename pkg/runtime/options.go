package runtime

import "github.com/magomedcoder/gogguf/pkg/model"

// Options задаёт параметры загрузки inference-движка
type Options struct {
	NGL       int // число transformer-слоёв на GPU (флаг -ngl)
	GPUMaxSeq int // макс. длина GPU KV-cache (0 = авто, до 4096)
}

func (o Options) modelOpts() model.Options {
	return model.Options{
		NGL:       o.NGL,
		GPUMaxSeq: o.GPUMaxSeq,
	}
}
