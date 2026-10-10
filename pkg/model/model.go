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

	switch arch {
	case "qwen3":
		return qwen3.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, opts.NBatch)
	case "qwen3moe":
		return qwen3.LoadMoE(store, opts.GPU, opts.NGL, opts.GPUMaxSeq, opts.NBatch)
	case "qwen2":
		// DeepSeek-R1-Distill-Qwen and other distill models with arch=qwen2 (no QK-norm)
		return mistral.LoadQwen2(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "qwen2moe":
		return mistral.LoadQwen2MoE(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "mistral":
		return mistral.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "gemma":
		return gemma.LoadGemma(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "gemma2":
		return gemma.LoadGemma2(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "phi2":
		return phi2.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "phi3":
		return phi3.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "deepseek":
		return deepseek.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "deepseek2":
		return deepseek2.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	case "llama":
		if isMixtralModel(r) || isMistralModel(r) {
			return mistral.LoadLlamaMeta(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
		}
		return llama.Load(store, opts.GPU, opts.NGL, opts.GPUMaxSeq)
	default:
		return nil, fmt.Errorf("model: architecture %q not supported", arch)
	}
}
