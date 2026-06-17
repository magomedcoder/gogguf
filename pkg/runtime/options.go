package runtime

import "github.com/magomedcoder/gguf.go/pkg/model"

// Options задаёт параметры загрузки inference-движка
type Options struct {
	NGL int // число transformer-слоёв на GPU (флаг -ngl)
}

func (o Options) modelOpts() model.Options {
	return model.Options{
		NGL: o.NGL,
	}
}
