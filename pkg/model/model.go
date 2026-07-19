package model

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/llama"
	"github.com/magomedcoder/gogguf/pkg/model/mistral"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - интерфейс архитектуры для forward pass
type Model interface {
	Forward(tokenIDs []int, startPos int) ([]float32, error)
	ResetCache()
}

// Load загружает модель по полю general.architecture
func Load(r *format.Reader, opts Options) (Model, error) {
	if err := opts.Normalize(); err != nil {
		return nil, err
	}

	arch, err := r.Metadata.String("general.architecture")
	if err != nil {
		return nil, err
	}

	store := weights.New(r)

	switch arch {
	case "qwen3":
		return qwen3.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "mistral":
		return mistral.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "llama":
		if isMistralModel(r) {
			return mistral.LoadLlamaMeta(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
		}
		return llama.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	default:
		return nil, fmt.Errorf("model: архитектура %q не поддерживается", arch)
	}
}
