package gemma

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerNorms struct {
	attnNorm     []float32
	ffnNorm      []float32
	attnPostNorm []float32 // gemma2
	ffnPostNorm  []float32 // gemma2
}

func loadNormWeights(w *weights.Store, cfg Config) ([]layerNorms, []float32, error) {
	layers := make([]layerNorms, cfg.NumLayers)
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if layers[i].attnNorm, err = w.Floats(p + "attn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("gemma: blk.%d attn_norm: %w", i, err)
		}

		if layers[i].ffnNorm, err = w.Floats(p + "ffn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("gemma: blk.%d ffn_norm: %w", i, err)
		}

		if cfg.Variant == VariantGemma2 {
			if layers[i].attnPostNorm, err = w.Floats(p + "post_attention_norm.weight"); err != nil {
				return nil, nil, fmt.Errorf("gemma: blk.%d post_attention_norm: %w", i, err)
			}

			if layers[i].ffnPostNorm, err = w.Floats(p + "post_ffw_norm.weight"); err != nil {
				return nil, nil, fmt.Errorf("gemma: blk.%d post_ffw_norm: %w", i, err)
			}
		}
	}

	outNorm, err := w.Floats("output_norm.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("gemma: output_norm: %w", err)
	}

	return layers, outNorm, nil
}
