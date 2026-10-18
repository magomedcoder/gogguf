package metal

// KernelsImplemented is false until Metal Shading Language kernels back Backend.
// pkg/model.Options.Normalize uses Backend.Ready (this flag) to decide whether darwin -ngl may offload.
const KernelsImplemented = false

// KernelPlan describes one planned Metal compute kernel (nothing is compiled or dispatched yet)
type KernelPlan struct {
	Name    string   // MSL function name
	Entries []string // Backend methods that dispatch it
	Notes   string
}

// PlannedKernels is the roadmap for a Metal backend, mirroring the CUDA fused set in pkg/device/cuda (not implemented).
// Order is the intended implementation order: decode matvec first (tg), then ops and residency, then batched GEMM (pp).
var PlannedKernels = []KernelPlan{
	{Name: "matmul_vec_f32", Entries: []string{"MatMulVec", "MatMulVecCached"}, Notes: "dense FP32/FP16 matvec, SIMD-group reduction"},
	{Name: "matmul_vec_q8_0", Entries: []string{"MatMulVecQ8_0Cached", "FFNSwiGLUQ8_0Cached"}, Notes: "Q8_0 block dequant fused into dot product"},
	{Name: "matmul_vec_q4_0", Entries: []string{"MatMulVecQ4_0Cached"}, Notes: "Q4_0 nibbles"},
	{Name: "matmul_vec_q4_k", Entries: []string{"MatMulVecQ4_KCached"}, Notes: "Q4_K super-blocks"},
	{Name: "matmul_vec_q5_k", Entries: []string{"MatMulVecQ5_KCached"}, Notes: "Q5_K super-blocks"},
	{Name: "matmul_vec_q6_k", Entries: []string{"MatMulVecQ6_KCached"}, Notes: "Q6_K super-blocks"},
	{Name: "matmul_vec_q4_1", Entries: []string{"MatMulVecQ4_1Cached"}, Notes: "Q4_1 (scale + min)"},
	{Name: "matmul_vec_q5_0", Entries: []string{"MatMulVecQ5_0Cached"}, Notes: "Q5_0 high-bit plane"},
	{Name: "matmul_vec_q5_1", Entries: []string{"MatMulVecQ5_1Cached"}, Notes: "Q5_1 (scale + min)"},
	{Name: "matmul_vec_q2_k", Entries: []string{"MatMulVecQ2_KCached"}, Notes: "Q2_K super-blocks"},
	{Name: "matmul_vec_q3_k", Entries: []string{"MatMulVecQ3_KCached"}, Notes: "Q3_K super-blocks"},
	{Name: "matmul_vec_q8_k", Entries: []string{"MatMulVecQ8_KCached"}, Notes: "Q8_K super-blocks"},
	{Name: "rmsnorm", Entries: []string{"RMSNormInto", "AttnFFNResidualQuantCached", "LogitsFromDevice"}, Notes: "threadgroup reduction"},
	{Name: "rope_heads", Entries: []string{"ApplyRoPEHeads", "QKVRoPEAttentionQuantCached"}, Notes: "NeoX layout"},
	{Name: "rope_heads_norm", Entries: []string{"ApplyRoPEHeadsNorm", "QKVRoPEAttentionQuantCached"}, Notes: "Llama NORM layout"},
	{Name: "swiglu", Entries: []string{"SwiGLUInPlace", "FFNSwiGLUQuantCached"}, Notes: "SiLU(gate)*up; GeGLU variant for FFNGeGLUQuantCached"},
	{Name: "attn_qk_softmax_v", Entries: []string{"AttentionScoresInto", "AttentionScoresKV"}, Notes: "GQA attention over GPU KV cache; flash-style fused variant later"},
	{Name: "kv_append", Entries: []string{"KVCacheInit", "KVCacheAppend", "KVCacheAppendN"}, Notes: "unified-memory MTLBuffer KV per layer"},
	{Name: "hidden_resident", Entries: []string{"HiddenUpload", "HiddenDownload", "HiddenResident"}, Notes: "residual stream kept in a MTLBuffer between layers"},
	{Name: "matmul_mat_q8_0", Entries: []string{"MatMulVecQ8_0Cached"}, Notes: "batched GEMM for prefill (-b), via simdgroup_matrix"},
}

// plannedKernelFor returns the first planned kernel dispatched by a Backend entry point ("" when none is named)
func plannedKernelFor(entry string) string {
	for _, k := range PlannedKernels {
		for _, e := range k.Entries {
			if e == entry {
				return k.Name
			}
		}
	}

	return ""
}
