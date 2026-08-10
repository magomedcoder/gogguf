package phi2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerNorms struct {
	attnW []float32
	attnB []float32
}

type outNorms struct {
	w []float32
	b []float32
}

func loadNormWeights(w *weights.Store, numLayers int) ([]layerNorms, outNorms, error) {
	layers := make([]layerNorms, numLayers)
	for i := range numLayers {
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if layers[i].attnW, err = w.Floats(p + "attn_norm.weight"); err != nil {
			return nil, outNorms{}, fmt.Errorf("phi2: blk.%d attn_norm.weight: %w", i, err)
		}

		if layers[i].attnB, err = w.Floats(p + "attn_norm.bias"); err != nil {
			return nil, outNorms{}, fmt.Errorf("phi2: blk.%d attn_norm.bias: %w", i, err)
		}
	}

	var out outNorms
	var err error
	if out.w, err = w.Floats("output_norm.weight"); err != nil {
		return nil, outNorms{}, fmt.Errorf("phi2: output_norm.weight: %w", err)
	}

	if out.b, err = w.Floats("output_norm.bias"); err != nil {
		return nil, outNorms{}, fmt.Errorf("phi2: output_norm.bias: %w", err)
	}

	return layers, out, nil
}
