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

// B2Kernels are the MSL entry points added for residency / fused decode and the native 4-bit matvecs (each must also be named in PlannedKernels).
// The fused decode path adds no kernel of its own: it encodes B1 kernels (plus add and the multi-row rmsnorm launch) into one command buffer per call.
var B2Kernels = []string{
	"add",
	"matmul_vec_q4_0",
	"matmul_vec_q4_k",
}

// KernelPlan describes one planned Metal compute kernel (nothing is compiled or dispatched yet)
type KernelPlan struct {
	Name    string   // MSL function name
	Entries []string // Backend methods that dispatch it
	Notes   string
}

// PlannedKernels is the roadmap for a Metal backend, mirroring the CUDA fused set in pkg/device/cuda; B1Kernels, B2Kernels (and kv_append on the host) are implemented, the rest is planned.
// Order is the intended implementation order: decode matvec first (tg), then ops and residency, then batched GEMM (pp).
var PlannedKernels = []KernelPlan{
	{Name: "matmul_vec_f32", Entries: []string{"MatMulVec", "MatMulVecCached"}, Notes: "dense FP32/FP16 matvec, SIMD-group reduction"},
	{Name: "matmul_vec_q8_0", Entries: []string{"MatMulVecQ8_0Cached", "FFNSwiGLUQ8_0Cached"}, Notes: "Q8_0 block dequant fused into dot product"},
	{Name: "matmul_vec_q4_0", Entries: []string{"MatMulVecQ4_0Cached"}, Notes: "B2: Q4_0 nibbles (pkg/quant layout); logits head only, not yet in the fused QKV / FFN path"},
	{Name: "matmul_vec_q4_k", Entries: []string{"MatMulVecQ4_KCached"}, Notes: "B2: Q4_K super-blocks (ggml layout); logits head only, not yet in the fused QKV / FFN path"},
	{Name: "matmul_vec_q5_k", Entries: []string{"MatMulVecQ5_KCached"}, Notes: "Q5_K super-blocks"},
	{Name: "matmul_vec_q6_k", Entries: []string{"MatMulVecQ6_KCached"}, Notes: "Q6_K super-blocks"},
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
	{Name: "rope_heads_norm", Entries: []string{"ApplyRoPEHeadsNorm", "QKVRoPEAttentionQuantCached"}, Notes: "Llama NORM layout"},
	{Name: "swiglu", Entries: []string{"SwiGLUInPlace", "FFNSwiGLUQuantCached", "AttnFFNResidualGeGLUQuantCached"}, Notes: "SiLU(gate)*up (B2: fused Q8_0 FFN); GeGLU variant for FFNGeGLUQuantCached / AttnFFNResidualGeGLUQuantCached stays planned"},
	{Name: "attn_qk_softmax_v", Entries: []string{"AttentionScoresInto", "AttentionScoresKV", "QKVRoPEAttentionQuantCached"}, Notes: "GQA attention over GPU KV cache; flash-style fused variant later"},
	{Name: "kv_append", Entries: []string{"KVCacheInit", "KVCacheAppend", "KVCacheAppendN", "KVCacheDropOldest", "KVCacheRebuildFromHost"}, Notes: "unified-memory MTLBuffer KV per layer"},
	{Name: "add", Entries: []string{"HiddenUpload", "HiddenDownload", "HiddenResident", "AttnFFNResidualQuantCached", "ResidAdd"}, Notes: "B2: a[i] += b[i] residual add; the residual stream lives in a MTLBuffer between layers (hidden_resident). ResidAdd (host delta, Phi-2) stays planned"},
	{Name: "matmul_mat_q8_0", Entries: []string{"MatMulVecQ8_0Cached"}, Notes: "batched GEMM for prefill (-b), via simdgroup_matrix"},
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
