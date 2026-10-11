package compute

import (
	"github.com/magomedcoder/gogguf/pkg/compute/attention"
	"github.com/magomedcoder/gogguf/pkg/compute/ffn"
	"github.com/magomedcoder/gogguf/pkg/compute/matmul"
	"github.com/magomedcoder/gogguf/pkg/compute/norm"
	"github.com/magomedcoder/gogguf/pkg/compute/rope"
)

// RoPE
type (
	RoPEMode  = rope.RoPEMode
	RoPEScale = rope.RoPEScale
)

const (
	RoPENeoX = rope.RoPENeoX
	RoPENorm = rope.RoPENorm
)

var (
	ApplyRoPE                   = rope.ApplyRoPE
	ApplyRoPEPartial            = rope.ApplyRoPEPartial
	ApplyRoPEPartialScaled      = rope.ApplyRoPEPartialScaled
	ApplyRoPENorm               = rope.ApplyRoPENorm
	ApplyRoPEHeads              = rope.ApplyRoPEHeads
	ApplyRoPEHeadsNorm          = rope.ApplyRoPEHeadsNorm
	ApplyRoPEHeadsMode          = rope.ApplyRoPEHeadsMode
	ApplyRoPEHeadsPartial       = rope.ApplyRoPEHeadsPartial
	ApplyRoPEHeadsPartialScaled = rope.ApplyRoPEHeadsPartialScaled
	RoPECosSin                  = rope.RoPECosSin
	RoPECosSinScaled            = rope.RoPECosSinScaled
	MaxRoPEPairs                = rope.MaxRoPEPairs
)

// Norm
var (
	RMSNorm          = norm.RMSNorm
	RMSNormInto      = norm.RMSNormInto
	RMSNormBatchInto = norm.RMSNormBatchInto
	LayerNormInto    = norm.LayerNormInto
	VectorRMS        = norm.VectorRMS
)

// FFN / elementwise
var (
	AddInPlace     = ffn.AddInPlace
	AddBiasInPlace = ffn.AddBiasInPlace
	ScaleInPlace   = ffn.ScaleInPlace
	SoftcapInPlace = ffn.SoftcapInPlace
	SwiGLUInPlace  = ffn.SwiGLUInPlace
	GELUInPlace    = ffn.GELUInPlace
	GeGLUInPlace   = ffn.GeGLUInPlace
	SiLU           = ffn.SiLU
)

// Matmul + embeddings
var (
	MatMulVec           = matmul.MatMulVec
	MatMulVecInto       = matmul.MatMulVecInto
	MatMulVecQ8_0       = matmul.MatMulVecQ8_0
	MatMulVecQ8_0Into   = matmul.MatMulVecQ8_0Into
	MatMulVecQ4_0       = matmul.MatMulVecQ4_0
	MatMulVecQ4_0Into   = matmul.MatMulVecQ4_0Into
	MatMulVecQ4_1       = matmul.MatMulVecQ4_1
	MatMulVecQ4_1Into   = matmul.MatMulVecQ4_1Into
	MatMulVecQ5_0       = matmul.MatMulVecQ5_0
	MatMulVecQ5_0Into   = matmul.MatMulVecQ5_0Into
	MatMulVecQ5_1       = matmul.MatMulVecQ5_1
	MatMulVecQ5_1Into   = matmul.MatMulVecQ5_1Into
	MatMulVecQ4_K       = matmul.MatMulVecQ4_K
	MatMulVecQ4_KInto   = matmul.MatMulVecQ4_KInto
	MatMulVecQ5_K       = matmul.MatMulVecQ5_K
	MatMulVecQ5_KInto   = matmul.MatMulVecQ5_KInto
	MatMulVecQ6_K       = matmul.MatMulVecQ6_K
	MatMulVecQ6_KInto   = matmul.MatMulVecQ6_KInto
	MatMulVecQ2_K       = matmul.MatMulVecQ2_K
	MatMulVecQ2_KInto   = matmul.MatMulVecQ2_KInto
	MatMulVecQ3_K       = matmul.MatMulVecQ3_K
	MatMulVecQ3_KInto   = matmul.MatMulVecQ3_KInto
	MatMulVecQ8_K       = matmul.MatMulVecQ8_K
	MatMulVecQ8_KInto   = matmul.MatMulVecQ8_KInto
	MatMulMatInto       = matmul.MatMulMatInto
	MatMulMatQ8_0Into   = matmul.MatMulMatQ8_0Into
	MatMulMatViaVecInto = matmul.MatMulMatViaVecInto
	MatMulColMajorInto  = attention.MatMulColMajorInto

	EmbeddingQ8_0     = matmul.EmbeddingQ8_0
	EmbeddingQ4_0     = matmul.EmbeddingQ4_0
	EmbeddingQ4_K     = matmul.EmbeddingQ4_K
	EmbeddingQ8_0Into = matmul.EmbeddingQ8_0Into
	EmbeddingQ4_0Into = matmul.EmbeddingQ4_0Into
	EmbeddingQ4_1Into = matmul.EmbeddingQ4_1Into
	EmbeddingQ5_0Into = matmul.EmbeddingQ5_0Into
	EmbeddingQ5_1Into = matmul.EmbeddingQ5_1Into
	EmbeddingQ4_KInto = matmul.EmbeddingQ4_KInto
	EmbeddingQ5_KInto = matmul.EmbeddingQ5_KInto
	EmbeddingQ6_KInto = matmul.EmbeddingQ6_KInto
	EmbeddingQ2_KInto = matmul.EmbeddingQ2_KInto
	EmbeddingQ3_KInto = matmul.EmbeddingQ3_KInto
	EmbeddingQ8_KInto = matmul.EmbeddingQ8_KInto
)

// Attention
var (
	AttentionScoresInto            = attention.AttentionScoresInto
	AttentionScoresIntoSoftcap     = attention.AttentionScoresIntoSoftcap
	AttentionScoresBatchCausalInto = attention.AttentionScoresBatchCausalInto
	AttentionMLAAbsorbedInto       = attention.AttentionMLAAbsorbedInto
	SoftmaxInPlace                 = attention.SoftmaxInPlace
)
