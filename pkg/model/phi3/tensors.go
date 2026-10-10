package phi3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerTensors struct {
	fusedQKV bool
	fusedFFN bool
	attnQKV  string
	attnQ    string
	attnK    string
	attnV    string
	attnOut  string
	ffnGate  string
	ffnUp    string
	ffnDown  string
}

func loadLayerTensors(w *weights.Store, numLayers int) ([]layerTensors, error) {
	layers := make([]layerTensors, numLayers)
	for i := range numLayers {
		p := fmt.Sprintf("blk.%d.", i)
		lt := layerTensors{
			attnOut: p + "attn_output.weight",
			ffnDown: p + "ffn_down.weight",
			ffnUp:   p + "ffn_up.weight",
		}

		qkv := p + "attn_qkv.weight"
		if _, err := w.Info(qkv); err == nil {
			lt.fusedQKV = true
			lt.attnQKV = qkv
		} else {
			lt.attnQ = p + "attn_q.weight"
			lt.attnK = p + "attn_k.weight"
			lt.attnV = p + "attn_v.weight"
			if _, err := w.Info(lt.attnQ); err != nil {
				return nil, fmt.Errorf("phi3: blk.%d: missing attn_qkv and attn_q", i)
			}
		}

		gate := p + "ffn_gate.weight"
		if _, err := w.Info(gate); err == nil {
			lt.ffnGate = gate
		} else {
			lt.fusedFFN = true
		}

		if _, err := w.Info(lt.ffnUp); err != nil {
			return nil, fmt.Errorf("phi3: blk.%d ffn_up: %w", i, err)
		}

		layers[i] = lt
	}

	return layers, nil
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

// loadRopeFactors loads LongRoPE short/long (optional; 4k models without tensors)
func loadRopeFactors(w *weights.Store) (short, long []float32, err error) {
	short, err = optionalFloats(w, "rope_factors_short.weight")
	if err != nil {
		return nil, nil, err
	}

	long, err = optionalFloats(w, "rope_factors_long.weight")
	if err != nil {
		return nil, nil, err
	}

	return short, long, nil
}

func optionalFloats(w *weights.Store, name string) ([]float32, error) {
	if _, err := w.Info(name); err != nil {
		return nil, nil
	}

	return w.Floats(name)
}
