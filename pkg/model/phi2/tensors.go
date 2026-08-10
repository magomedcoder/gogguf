package phi2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerTensors struct {
	fusedQKV bool
	attnQKV  string
	attnQKVb string // bias; optional
	attnQ    string
	attnK    string
	attnV    string
	attnQb   string
	attnKb   string
	attnVb   string
	attnOut  string
	attnOutB string
	ffnUp    string
	ffnUpB   string
	ffnDown  string
	ffnDownB string
}

func loadLayerTensors(w *weights.Store, numLayers int) ([]layerTensors, error) {
	layers := make([]layerTensors, numLayers)
	for i := range numLayers {
		p := fmt.Sprintf("blk.%d.", i)
		lt := layerTensors{
			attnOut:  p + "attn_output.weight",
			attnOutB: p + "attn_output.bias",
			ffnUp:    p + "ffn_up.weight",
			ffnUpB:   p + "ffn_up.bias",
			ffnDown:  p + "ffn_down.weight",
			ffnDownB: p + "ffn_down.bias",
		}

		qkv := p + "attn_qkv.weight"
		if _, err := w.Info(qkv); err == nil {
			lt.fusedQKV = true
			lt.attnQKV = qkv
			lt.attnQKVb = p + "attn_qkv.bias"
		} else {
			lt.attnQ = p + "attn_q.weight"
			lt.attnK = p + "attn_k.weight"
			lt.attnV = p + "attn_v.weight"
			lt.attnQb = p + "attn_q.bias"
			lt.attnKb = p + "attn_k.bias"
			lt.attnVb = p + "attn_v.bias"
			if _, err := w.Info(lt.attnQ); err != nil {
				return nil, fmt.Errorf("phi2: blk.%d: нет attn_qkv и attn_q", i)
			}
		}

		for _, name := range []string{lt.attnOut, lt.ffnUp, lt.ffnDown} {
			if _, err := w.Info(name); err != nil {
				return nil, fmt.Errorf("phi2: %s: %w", name, err)
			}
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

func optionalFloats(w *weights.Store, name string) ([]float32, error) {
	if _, err := w.Info(name); err != nil {
		return nil, nil
	}

	return w.Floats(name)
}
