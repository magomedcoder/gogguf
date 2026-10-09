package ops

const maxRoPEPairs = 128

// RoPEMode - RoPE pair layout within a head
type RoPEMode int

const (
	// RoPENeoX - pairs (i, i+head_dim/2): Qwen / NeoX / Gemma
	RoPENeoX RoPEMode = iota

	// RoPENorm - adjacent pairs (2i, 2i+1): Llama
	RoPENorm
)

// ApplyRoPEHeadsMode applies RoPE in the selected mode
func ApplyRoPEHeadsMode(mode RoPEMode, v []float32, nHeads, headDim, pos int, freqBase float32) {
	if mode == RoPENorm {
		ApplyRoPEHeadsNorm(v, nHeads, headDim, pos, freqBase)
		return
	}

	ApplyRoPEHeads(v, nHeads, headDim, pos, freqBase)
}

// MaxRoPEPairs maximum headDim/2 for batched RoPE on CPU/GPU
func MaxRoPEPairs() int {
	return maxRoPEPairs
}

// ApplyRoPEHeads applies NeoX RoPE to nHeads heads in v; sin/cos computed once per position
func ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) {
	ApplyRoPEHeadsPartial(v, nHeads, headDim, headDim, pos, freqBase)
}

// ApplyRoPEHeadsPartial applies NeoX RoPE to first nRot dims of each head (Phi-3 partial rotary)
func ApplyRoPEHeadsPartial(v []float32, nHeads, headDim, nRot, pos int, freqBase float32) {
	ApplyRoPEHeadsPartialScaled(v, nHeads, headDim, nRot, pos, freqBase, RoPEScale{})
}

// ApplyRoPEHeadsPartialScaled like ApplyRoPEHeadsPartial + LongRoPE factors
func ApplyRoPEHeadsPartialScaled(v []float32, nHeads, headDim, nRot, pos int, freqBase float32, scale RoPEScale) {
	if nHeads <= 0 || headDim <= 0 {
		return
	}

	if nRot <= 0 || nRot > headDim {
		nRot = headDim
	}

	if nRot%2 != 0 {
		nRot--
	}

	half := nRot / 2
	if half > maxRoPEPairs {
		for h := range nHeads {
			off := h * headDim
			ApplyRoPEPartialScaled(v[off:off+headDim], pos, freqBase, nRot, scale)
		}
		return
	}

	var cosTab [maxRoPEPairs]float32
	var sinTab [maxRoPEPairs]float32

	RoPECosSinScaled(cosTab[:half], sinTab[:half], nRot, pos, freqBase, scale)

	for h := range nHeads {
		base := h * headDim
		for i := range half {
			x0 := float64(v[base+i])
			x1 := float64(v[base+half+i])
			c, s := float64(cosTab[i]), float64(sinTab[i])
			v[base+i] = float32(x0*c - x1*s)
			v[base+half+i] = float32(x0*s + x1*c)
		}
	}
}
