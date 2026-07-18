package mistral

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerTensors struct {
	attnQ   string
	attnK   string
	attnV   string
	attnOut string
	ffnGate string
	ffnUp   string
	ffnDown string
}

func loadLayerTensors(numLayers int) []layerTensors {
	layers := make([]layerTensors, numLayers)
	for i := range numLayers {
		p := fmt.Sprintf("blk.%d.", i)
		layers[i] = layerTensors{
			attnQ:   p + "attn_q.weight",
			attnK:   p + "attn_k.weight",
			attnV:   p + "attn_v.weight",
			attnOut: p + "attn_output.weight",
			ffnGate: p + "ffn_gate.weight",
			ffnUp:   p + "ffn_up.weight",
			ffnDown: p + "ffn_down.weight",
		}
	}
	return layers
}

func resolveLMHeadName(w *weights.Store) (string, error) {
	const primary = "output.weight"
	if _, err := w.Info(primary); err == nil {
		return primary, nil
	}

	const fallback = "token_embd.weight"
	if _, err := w.Info(fallback); err != nil {
		return "", err
	}

	return fallback, nil
}
