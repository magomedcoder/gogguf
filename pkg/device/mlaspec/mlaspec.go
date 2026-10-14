package mlaspec

// Params describes absorbed MLA (DeepSeek-V2/V3) attention sizes.
//
// K cache row = kv_lora + rope (kv_cmpr, k_pe), V cache row = kv_lora (kv_cmpr); one shared KV head.
type Params struct {
	NHeads int
	Nope   int // qk_nope_head_dim (q_nope width, k_b inner dim)
	KVLora int
	Rope   int
	VHead  int
	Scale  float32 // kq scale applied to q.k
}

// KDim is the K cache row width.
func (p Params) KDim() int {
	return p.KVLora + p.Rope
}

// QDim is the width of q fed to MLAAttention: [NHeads][Nope+Rope].
func (p Params) QDim() int {
	return p.NHeads * (p.Nope + p.Rope)
}

// OutDim is the attention output width: [NHeads][VHead].
func (p Params) OutDim() int {
	return p.NHeads * p.VHead
}

// WeightLoader returns FP32 host weights by tensor name.
// It is only called when the backend does not already hold a device copy, so the hot path never touches host memory for k_b/v_b.
type WeightLoader func(name string) ([]float32, error)
