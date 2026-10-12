package model

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model/deepseek"
	"github.com/magomedcoder/gogguf/pkg/model/deepseek2"
	"github.com/magomedcoder/gogguf/pkg/model/gemma"
	"github.com/magomedcoder/gogguf/pkg/model/llama"
	"github.com/magomedcoder/gogguf/pkg/model/mistral"
	"github.com/magomedcoder/gogguf/pkg/model/phi2"
	"github.com/magomedcoder/gogguf/pkg/model/phi3"
	"github.com/magomedcoder/gogguf/pkg/model/qwen2"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Model - architecture interface for forward pass
type Model interface {
	Forward(tokenIDs []int, startPos int) ([]float32, error)

	// Embed - last-token hidden after output-norm (before lm_head); clears KV-cache
	Embed(tokenIDs []int) ([]float32, error)

	EmbeddingDim() int

	ResetCache()

	Close() error
}

// Load loads a model from general.architecture
func Load(r *format.Reader, opts Options) (Model, error) {
	if err := opts.Normalize(); err != nil {
		return nil, err
	}

	arch, err := r.Metadata.String("general.architecture")
	if err != nil {
		return nil, err
	}

	store := weights.New(r)
	nBatch := ClampNBatch(opts.NBatch)

	switch arch {
	case "qwen3":
		return qwen3.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "qwen3moe":
		return qwen3.LoadMoE(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "qwen2":
		return qwen2.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "qwen2moe":
		return qwen2.LoadMoE(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "mistral":
		return mistral.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "gemma":
		return gemma.LoadGemma(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "gemma2":
		return gemma.LoadGemma2(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "phi2":
		return phi2.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "phi3":
		return phi3.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "deepseek":
		return deepseek.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "deepseek2":
		return deepseek2.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	case "llama":
		if isMixtralModel(r) || isMistralModel(r) {
			return mistral.LoadLlamaMeta(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
		}
		return llama.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, nBatch)
	default:
		return nil, fmt.Errorf("model: architecture %q not supported", arch)
	}
}
