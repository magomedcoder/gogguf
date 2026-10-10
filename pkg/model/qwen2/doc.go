// Package qwen2 provides loaders for general.architecture=qwen2 and qwen2moe.
// Implementation lives in pkg/model/mistral (shared NeoX / MoE topology).
package qwen2

import (
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model/mistral"
	"github.com/magomedcoder/gogguf/pkg/weights"
)

// Load loads a qwen2 GGUF model.
func Load(w *weights.Store, g gpu.Backend, ngl, gpuMaxSeq int) (*mistral.Model, error) {
	return mistral.LoadQwen2(w, g, ngl, gpuMaxSeq)
}

// LoadMoE loads a qwen2moe GGUF model.
func LoadMoE(w *weights.Store, g gpu.Backend, ngl, gpuMaxSeq int) (*mistral.Model, error) {
	return mistral.LoadQwen2MoE(w, g, ngl, gpuMaxSeq)
}
