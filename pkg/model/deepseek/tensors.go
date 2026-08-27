package deepseek

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerNorms struct {
	attnNorm []float32
	ffnNorm  []float32
}

type layerTensors struct {
	attnQ   string
	attnK   string
	attnV   string
	attnOut string
	// dense
	ffnGate string
	ffnUp   string
	ffnDown string
	// MoE
	moe       bool
	gateInp   string
	gateExps  string
	upExps    string
	downExps  string
	gateShexp string
	upShexp   string
	downShexp string
}

func loadNormWeights(w *weights.Store, numLayers int) ([]layerNorms, []float32, error) {
	layers := make([]layerNorms, numLayers)
	for i := range numLayers {
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if layers[i].attnNorm, err = w.Floats(p + "attn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("deepseek: blk.%d attn_norm: %w", i, err)
		}

		if layers[i].ffnNorm, err = w.Floats(p + "ffn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("deepseek: blk.%d ffn_norm: %w", i, err)
		}
	}

	outNorm, err := w.Floats("output_norm.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("deepseek: output_norm: %w", err)
	}

	return layers, outNorm, nil
}

func loadLayerTensors(w *weights.Store, cfg Config) ([]layerTensors, error) {
	layers := make([]layerTensors, cfg.NumLayers)
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("blk.%d.", i)
		lt := layerTensors{
			attnQ:   p + "attn_q.weight",
			attnK:   p + "attn_k.weight",
			attnV:   p + "attn_v.weight",
			attnOut: p + "attn_output.weight",
		}

		if cfg.isMoELayer(i) {
			lt.moe = true
			lt.gateInp = p + "ffn_gate_inp.weight"
			lt.gateExps = p + "ffn_gate_exps.weight"
			lt.upExps = p + "ffn_up_exps.weight"
			lt.downExps = p + "ffn_down_exps.weight"
			for _, name := range []string{lt.gateInp, lt.gateExps, lt.upExps, lt.downExps} {
				if _, err := w.Info(name); err != nil {
					return nil, fmt.Errorf("deepseek: %s: %w", name, err)
				}
			}

			if cfg.ExpertShared > 0 {
				lt.gateShexp = p + "ffn_gate_shexp.weight"
				lt.upShexp = p + "ffn_up_shexp.weight"
				lt.downShexp = p + "ffn_down_shexp.weight"
				for _, name := range []string{lt.gateShexp, lt.upShexp, lt.downShexp} {
					if _, err := w.Info(name); err != nil {
						return nil, fmt.Errorf("deepseek: %s: %w", name, err)
					}
				}
			}
		} else {
			lt.ffnGate = p + "ffn_gate.weight"
			lt.ffnUp = p + "ffn_up.weight"
			lt.ffnDown = p + "ffn_down.weight"
			for _, name := range []string{lt.ffnGate, lt.ffnUp, lt.ffnDown} {
				if _, err := w.Info(name); err != nil {
					return nil, fmt.Errorf("deepseek: %s: %w", name, err)
				}
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
