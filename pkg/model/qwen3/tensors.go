package qwen3

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerTensors struct {
	attnQ    string
	attnK    string
	attnV    string
	attnOut  string
	ffnGate  string
	ffnUp    string
	ffnDown  string
	moe      bool
	gateInp  string
	gateExps string
	upExps   string
	downExps string
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
		if cfg.isMoE() {
			lt.moe = true
			lt.gateInp = p + "ffn_gate_inp.weight"
			lt.gateExps = p + "ffn_gate_exps.weight"
			lt.upExps = p + "ffn_up_exps.weight"
			lt.downExps = p + "ffn_down_exps.weight"
			for _, name := range []string{lt.gateInp, lt.gateExps, lt.upExps, lt.downExps} {
				if _, err := w.Info(name); err != nil {
					return nil, fmt.Errorf("qwen3: %s: %w", name, err)
				}
			}
		} else {
			lt.ffnGate = p + "ffn_gate.weight"
			lt.ffnUp = p + "ffn_up.weight"
			lt.ffnDown = p + "ffn_down.weight"
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
