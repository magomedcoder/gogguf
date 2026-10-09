package sampler

import (
	"math/rand/v2"
	"slices"
	"sort"
)

// Config - sampling parameters
type Config struct {
	Temp float32 // 0 = greedy
	TopK int     // 0 = disabled
	TopP float32 // 1.0 = disabled
	MinP float32 // 0 = disabled
	Seed uint64
}

// New creates sampler from configuration
func New(cfg Config) Func {
	if cfg.Temp <= 0 && cfg.TopK <= 0 && (cfg.TopP <= 0 || cfg.TopP >= 1) {
		return Greedy
	}

	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x9E3779B97F4A7C15))

	return func(logits []float32) int {
		if len(logits) == 0 {
			return -1
		}

		if cfg.Temp <= 0 && cfg.TopK <= 0 && (cfg.TopP <= 0 || cfg.TopP >= 1) {
			return Greedy(logits)
		}

		scaled := make([]float32, len(logits))
		copy(scaled, logits)
		applyTemperature(scaled, cfg.Temp)

		mask := make([]bool, len(scaled))
		for i := range mask {
			mask[i] = true
		}

		if cfg.TopK > 0 && cfg.TopK < len(scaled) {
			applyTopK(scaled, mask, cfg.TopK)
		}

		if cfg.MinP > 0 {
			ApplyMinP(scaled, cfg.MinP)
			for i, v := range scaled {
				if v <= -1e9 {
					mask[i] = false
				}
			}
		}

		for i, ok := range mask {
			if !ok {
				scaled[i] = -1e10
			}
		}

		probs := softmaxInPlace(scaled)

		if cfg.TopP > 0 && cfg.TopP < 1 {
			applyTopP(probs, mask, cfg.TopP)
		}

		return sampleMasked(probs, mask, rng)
	}
}

type idxProb struct {
	i    int
	prob float64
}

func applyTopK(logits []float32, mask []bool, k int) {
	items := make([]idxProb, 0, len(logits))
	for i, v := range logits {
		if mask[i] {
			items = append(items, idxProb{i: i, prob: float64(v)})
		}
	}

	sort.Slice(items, func(a, b int) bool {
		return items[a].prob > items[b].prob
	})

	if k > len(items) {
		k = len(items)
	}

	keep := make(map[int]struct{}, k)
	for i := 0; i < k; i++ {
		keep[items[i].i] = struct{}{}
	}

	for i := range mask {
		if _, ok := keep[i]; !ok {
			mask[i] = false
		}
	}
}

func applyTopP(probs []float64, mask []bool, p float32) {
	items := make([]idxProb, 0, len(probs))
	for i, prob := range probs {
		if mask[i] {
			items = append(items, idxProb{i: i, prob: prob})
		}
	}

	sort.Slice(items, func(a, b int) bool {
		return items[a].prob > items[b].prob
	})

	cum := 0.0
	keep := make(map[int]struct{}, len(items))
	for j, it := range items {
		keep[it.i] = struct{}{}
		cum += it.prob
		if cum >= float64(p)-1e-6 {
			break
		}
		_ = j
	}

	for i := range mask {
		if _, ok := keep[i]; !ok {
			mask[i] = false
		}
	}
}

func sampleMasked(probs []float64, mask []bool, rng *rand.Rand) int {
	sum := 0.0
	for i, ok := range mask {
		if ok {
			sum += probs[i]
		}
	}

	if sum <= 0 {
		return Greedy(float32SliceFromProbs(probs))
	}

	r := rng.Float64() * sum
	for i, ok := range mask {
		if !ok {
			continue
		}

		r -= probs[i]
		if r <= 0 {
			return i
		}
	}

	for i, m := range slices.Backward(mask) {
		if m {
			return i
		}
	}

	return 0
}

func float32SliceFromProbs(probs []float64) []float32 {
	out := make([]float32, len(probs))
	for i, p := range probs {
		out[i] = float32(p)
	}

	return out
}
