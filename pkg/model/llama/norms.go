package llama

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerNorms struct {
	attnNorm []float32
	ffnNorm  []float32
}

func loadNormWeights(w *weights.Store, numLayers int) ([]layerNorms, []float32, error) {
	layers := make([]layerNorms, numLayers)
	for i := range numLayers {
		p := fmt.Sprintf("blk.%d.", i)
		var err error

		if layers[i].attnNorm, err = w.Floats(p + "attn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("llama: blk.%d attn_norm: %w", i, err)
		}

		if layers[i].ffnNorm, err = w.Floats(p + "ffn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("llama: blk.%d ffn_norm: %w", i, err)
		}
	}

	outNorm, err := w.Floats("output_norm.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("llama: output_norm: %w", err)
	}

	return layers, outNorm, nil
}
