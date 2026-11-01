package prefill

import "fmt"

// Run drives multi-token prefill: sync -> embed -> per-layer block -> advance -> optional logits.
// Arch-specific SWA/GeGLU/MoE/RoPE stay inside block/logits callbacks.
func Run(arch string, nBatch, numLayers, startPos int, tokenIDs []int, needLogits bool, syncHidden func() error, embed func(batch int) error, block func(layer, startPos, batch int) error, advance func(batch int), logits func(batch int) error) error {
	b := len(tokenIDs)
	if b == 0 {
		return fmt.Errorf("%s: empty batch", arch)
	}

	if b > nBatch {
		return fmt.Errorf("%s: batch=%d > n_batch=%d", arch, b, nBatch)
	}

	if syncHidden != nil {
		if err := syncHidden(); err != nil {
			return err
		}
	}

	if err := embed(b); err != nil {
		return err
	}

	for layer := range numLayers {
		if err := block(layer, startPos, b); err != nil {
			return err
		}
	}

	advance(b)
	if !needLogits {
		return nil
	}

	if logits == nil {
		return nil
	}

	return logits(b)
}
