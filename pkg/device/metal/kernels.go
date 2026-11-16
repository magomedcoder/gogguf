package metal

import "slices"

// KernelsImplemented (kernels_impl_*.go) is true only in a darwin + cgo build with -tags metal, where the MSL kernels in msl.go back Backend; false everywhere else.
// pkg/model.Options.Normalize uses Backend.Ready (this flag plus an opened runtime) to decide whether darwin -ngl may offload.

// B1Kernels are the MSL entry points compiled by the darwin runtime for the B1 decode path (each must also be named in PlannedKernels).
// kv_append has no kernel: unified-memory KV buffers are written from Go with a plain memcpy.
var B1Kernels = []string{
	"matmul_vec_f32",
	"matmul_vec_q8_0",
	"rmsnorm",
	"rope_heads",
	"swiglu",
	"attn_qk_softmax_v",
}

// B2Kernels are the MSL entry points added for residency / fused decode, the native Q4_0 / Q4_K / Q5_K / Q6_K matvecs and the Llama NORM RoPE (each must also be named in PlannedKernels).
// The fused decode path adds no kernel of its own: it encodes B1 kernels (plus add, the multi-row rmsnorm launch and the matmul_vec_* / rope_heads* kernel matching the weight type / RoPE mode) into one command buffer per call.
var B2Kernels = []string{
	"add",
	"matmul_vec_q4_0",
	"matmul_vec_q4_k",
	"matmul_vec_q5_k",
	"matmul_vec_q6_k",
	"rope_heads_norm",
}

// B3Kernels are the MSL entry points added for prefill (-b): the batched Q8_0 GEMM (each must also be named in PlannedKernels).
// Batch attention and the batch FFN add no kernel of their own: they encode attn_qk_softmax_v / matmul_mat_q8_0 + swiglu launches into one command buffer per call.
var B3Kernels = []string{
	"matmul_mat_q8_0",
}

// KernelPlan describes one planned Metal compute kernel (nothing is compiled or dispatched yet)
type KernelPlan struct {
	Name    string   // MSL function name
	Entries []string // Backend methods that dispatch it
	Notes   string
}

// PlannedKernels is the roadmap for a Metal backend, mirroring the CUDA fused set in pkg/device/cuda; B1Kernels, B2Kernels, B3Kernels (and kv_append on the host) are implemented, the rest is planned.
// Order is the intended implementation order: decode matvec first (tg), then ops and residency, then batched GEMM (pp).
var PlannedKernels = []KernelPlan{
	{Name: "matmul_vec_f32", Entries: []string{"MatMulVec", "MatMulVecCached"}, Notes: "dense FP32/FP16 matvec, SIMD-group reduction"},
	{Name: "matmul_vec_q8_0", Entries: []string{"MatMulVecQ8_0Cached", "FFNSwiGLUQ8_0Cached"}, Notes: "Q8_0 block dequant fused into dot product; also the fused QKV / attn+FFN / FFN / logits matvec for Q8_0 weights"},
	{Name: "matmul_vec_q4_0", Entries: []string{"MatMulVecQ4_0Cached"}, Notes: "B2: Q4_0 nibbles (pkg/quant layout); logits head and fused QKV / attn+FFN / FFN path (dtype.Q4_0)"},
	{Name: "matmul_vec_q4_k", Entries: []string{"MatMulVecQ4_KCached"}, Notes: "B2: Q4_K super-blocks (ggml layout); logits head and fused QKV / attn+FFN / FFN path (dtype.Q4_K)"},
	{Name: "matmul_vec_q5_k", Entries: []string{"MatMulVecQ5_KCached"}, Notes: "B2: Q5_K super-blocks (packed 176-byte pkg/quant / ggml layout, not the CUDA expanded 224); logits head and fused path (dtype.Q5_K)"},
	{Name: "matmul_vec_q6_k", Entries: []string{"MatMulVecQ6_KCached"}, Notes: "B2: Q6_K super-blocks (packed 210-byte pkg/quant / ggml layout, not the CUDA expanded 256); logits head and fused path (dtype.Q6_K)"},
	{Name: "matmul_vec_q4_1", Entries: []string{"MatMulVecQ4_1Cached"}, Notes: "Q4_1 (scale + min)"},
	{Name: "matmul_vec_q5_0", Entries: []string{"MatMulVecQ5_0Cached"}, Notes: "Q5_0 high-bit plane"},
	{Name: "matmul_vec_q5_1", Entries: []string{"MatMulVecQ5_1Cached"}, Notes: "Q5_1 (scale + min)"},
	{Name: "matmul_vec_q2_k", Entries: []string{"MatMulVecQ2_KCached"}, Notes: "Q2_K super-blocks"},
	{Name: "matmul_vec_q3_k", Entries: []string{"MatMulVecQ3_KCached"}, Notes: "Q3_K super-blocks"},
	{Name: "matmul_vec_q8_k", Entries: []string{"MatMulVecQ8_KCached"}, Notes: "Q8_K super-blocks"},
	{Name: "rmsnorm", Entries: []string{"RMSNormInto", "QKVRoPEAttentionQuantCached", "AttnFFNResidualQuantCached", "LogitsFromDevice"}, Notes: "threadgroup reduction; B2 launches it per row (rows = nHeads for QK-norm) and on the device-resident residual"},
	{Name: "layernorm", Entries: []string{"LayerNormBiasInto", "LayerNormResidInto", "LogitsFromDeviceLayerNorm"}, Notes: "mean/var + bias (Phi-2)"},
	{Name: "gelu", Entries: []string{"GELUInPlace"}, Notes: "tanh-approx GELU (Phi-2 FFN)"},
	{Name: "rope_heads", Entries: []string{"ApplyRoPEHeads", "ApplyRoPEHeadsScaled", "QKVRoPEAttentionQuantCached"}, Notes: "NeoX layout; Scaled = YaRN/LongRoPE tables"},
	{Name: "rope_heads_norm", Entries: []string{"ApplyRoPEHeadsNorm", "QKVRoPEAttentionQuantCached"}, Notes: "B2: Llama NORM layout, adjacent pairs (2i, 2i+1), same host cos/sin table as rope_heads; fused QKV picks it for compute.RoPENorm"},
	{Name: "swiglu", Entries: []string{"SwiGLUInPlace", "FFNSwiGLUQuantCached", "FFNSwiGLUMatQuantCached", "AttnFFNResidualGeGLUQuantCached"}, Notes: "SiLU(gate)*up (B2: fused FFN for Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K; B3: batch FFN, whole [batch][ffn] in one launch); GeGLU variant for FFNGeGLUQuantCached / AttnFFNResidualGeGLUQuantCached stays planned"},
	{Name: "attn_qk_softmax_v", Entries: []string{"AttentionScoresInto", "AttentionScoresKV", "AttentionScoresKVBatch", "QKVRoPEAttentionQuantCached"}, Notes: "GQA attention over GPU KV cache; B3 batch = one launch per query row (causal seqLen = pastLen+i+1) encoded into one command buffer; flash-style fused variant later"},
	{Name: "kv_append", Entries: []string{"KVCacheInit", "KVCacheAppend", "KVCacheAppendN", "KVCacheDropOldest", "KVCacheRebuildFromHost"}, Notes: "unified-memory MTLBuffer KV per layer"},
	{Name: "add", Entries: []string{"HiddenUpload", "HiddenDownload", "HiddenResident", "AttnFFNResidualQuantCached", "ResidAdd"}, Notes: "B2: a[i] += b[i] residual add; the residual stream lives in a MTLBuffer between layers (hidden_resident). ResidAdd (host delta, Phi-2) stays planned"},
	{Name: "matmul_mat_q8_0", Entries: []string{"MatMulMatQ8_0CachedInto", "FFNSwiGLUMatQuantCached"}, Notes: "B3: batched Q8_0 GEMM for prefill (-b), CUDA-parity thread-per-(batch,row) output, one launch for W*X[batch]; simdgroup_matrix tiled GEMM is a follow-up (needs a Mac to tune); other quants / F32 loop the matvec kernels inside MatMulMat*"},
}

// plannedKernelFor returns the first planned kernel dispatched by a Backend entry point ("" when none is named)
func plannedKernelFor(entry string) string {
	for _, k := range PlannedKernels {
		if slices.Contains(k.Entries, entry) {
			return k.Name
		}
	}

	return ""
}
