package ops

const maxRoPEPairs = 128

// MaxRoPEPairs максимальный headDim/2 для batched RoPE на CPU/GPU
func MaxRoPEPairs() int {
	return maxRoPEPairs
}

// ApplyRoPEHeads применяет NeoX RoPE к nHeads головам в v; sin/cos вычисляются один раз на позицию
func ApplyRoPEHeads(v []float32, nHeads, headDim, pos int, freqBase float32) {
	ApplyRoPEHeadsPartial(v, nHeads, headDim, headDim, pos, freqBase)
}

// ApplyRoPEHeadsPartial применяет NeoX RoPE к первым nRot dim каждой головы (Phi-3 partial rotary)
func ApplyRoPEHeadsPartial(v []float32, nHeads, headDim, nRot, pos int, freqBase float32) {
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
	if half > maxRoPEPairs || half != headDim/2 {
		for h := range nHeads {
			off := h * headDim
			ApplyRoPEPartial(v[off:off+headDim], pos, freqBase, nRot)
		}
		return
	}

	var cosTab [maxRoPEPairs]float32
	var sinTab [maxRoPEPairs]float32

	RoPECosSin(cosTab[:half], sinTab[:half], nRot, pos, freqBase)

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
