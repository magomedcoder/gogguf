package deepseek2

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/weights"
)

type layerNorms struct {
	attnNorm []float32
	ffnNorm  []float32
	qANorm   []float32 // optional (not lite)
	kvANorm  []float32
}

type layerTensors struct {
	// Q projection
	liteQ bool
	q     string // attn_q.weight (lite)
	qA    string
	qB    string
	// KV MLA
	kvAMQA   string
	kB       string // attn_k_b.weight (absorbed)
	vB       string // attn_v_b.weight
	kvB      string // legacy attn_kv_b.weight
	absorbed bool
	attnOut  string
	// FFN
	ffnGate   string
	ffnUp     string
	ffnDown   string
	moe       bool
	gateInp   string
	gateExps  string
	upExps    string
	downExps  string
	gateShexp string
	upShexp   string
	downShexp string
	expProbsB string // optional router bias
}

func loadNormWeights(w *weights.Store, cfg Config) ([]layerNorms, []float32, error) {
	layers := make([]layerNorms, cfg.NumLayers)
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if layers[i].attnNorm, err = w.Floats(p + "attn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("deepseek2: %sattn_norm: %w", p, err)
		}

		if layers[i].ffnNorm, err = w.Floats(p + "ffn_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("deepseek2: %sffn_norm: %w", p, err)
		}

		if layers[i].kvANorm, err = w.Floats(p + "attn_kv_a_norm.weight"); err != nil {
			return nil, nil, fmt.Errorf("deepseek2: %sattn_kv_a_norm: %w", p, err)
		}

		if cfg.QLoraRank > 0 {
			if layers[i].qANorm, err = w.Floats(p + "attn_q_a_norm.weight"); err != nil {
				return nil, nil, fmt.Errorf("deepseek2: %sattn_q_a_norm: %w", p, err)
			}
		}
	}

	outNorm, err := w.Floats("output_norm.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("deepseek2: output_norm: %w", err)
	}

	return layers, outNorm, nil
}

func loadLayerTensors(w *weights.Store, cfg Config) ([]layerTensors, error) {
	layers := make([]layerTensors, cfg.NumLayers)
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("blk.%d.", i)
		lt := layerTensors{attnOut: p + "attn_output.weight"}

		if cfg.QLoraRank > 0 {
			lt.qA = p + "attn_q_a.weight"
			lt.qB = p + "attn_q_b.weight"
			for _, n := range []string{lt.qA, lt.qB} {
				if _, err := w.Info(n); err != nil {
					return nil, fmt.Errorf("deepseek2: %s: %w", n, err)
				}
			}
		} else {
			lt.liteQ = true
			lt.q = p + "attn_q.weight"
			if _, err := w.Info(lt.q); err != nil {
				return nil, fmt.Errorf("deepseek2: %s: %w", lt.q, err)
			}
		}

		lt.kvAMQA = p + "attn_kv_a_mqa.weight"
		if _, err := w.Info(lt.kvAMQA); err != nil {
			return nil, fmt.Errorf("deepseek2: %s: %w", lt.kvAMQA, err)
		}

		kB, errKB := w.Info(p + "attn_k_b.weight")
		vB, errVB := w.Info(p + "attn_v_b.weight")
		if errKB == nil && errVB == nil {
			lt.absorbed = true
			lt.kB = p + "attn_k_b.weight"
			lt.vB = p + "attn_v_b.weight"
			_ = kB
			_ = vB
		} else {
			lt.kvB = p + "attn_kv_b.weight"
			if _, err := w.Info(lt.kvB); err != nil {
				return nil, fmt.Errorf("deepseek2: нужен attn_k_b/attn_v_b или attn_kv_b: %w", err)
			}
		}

		if _, err := w.Info(lt.attnOut); err != nil {
			return nil, fmt.Errorf("deepseek2: %s: %w", lt.attnOut, err)
		}

		if cfg.isMoELayer(i) {
			lt.moe = true
			lt.gateInp = p + "ffn_gate_inp.weight"
			lt.gateExps = p + "ffn_gate_exps.weight"
			lt.upExps = p + "ffn_up_exps.weight"
			lt.downExps = p + "ffn_down_exps.weight"

			for _, n := range []string{lt.gateInp, lt.gateExps, lt.upExps, lt.downExps} {
				if _, err := w.Info(n); err != nil {
					return nil, fmt.Errorf("deepseek2: %s: %w", n, err)
				}
			}

			if cfg.ExpertShared > 0 {
				lt.gateShexp = p + "ffn_gate_shexp.weight"
				lt.upShexp = p + "ffn_up_shexp.weight"
				lt.downShexp = p + "ffn_down_shexp.weight"
				for _, n := range []string{lt.gateShexp, lt.upShexp, lt.downShexp} {
					if _, err := w.Info(n); err != nil {
						return nil, fmt.Errorf("deepseek2: %s: %w", n, err)
					}
				}
			}

			if _, err := w.Info(p + "exp_probs_b.bias"); err == nil {
				lt.expProbsB = p + "exp_probs_b.bias"
			}
		} else {
			lt.ffnGate = p + "ffn_gate.weight"
			lt.ffnUp = p + "ffn_up.weight"
			lt.ffnDown = p + "ffn_down.weight"
			for _, n := range []string{lt.ffnGate, lt.ffnUp, lt.ffnDown} {
				if _, err := w.Info(n); err != nil {
					return nil, fmt.Errorf("deepseek2: %s: %w", n, err)
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
