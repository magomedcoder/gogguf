//go:build metal

package metal

import (
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

// Every B1/B2 kernel is defined in the embedded MSL (static check, runs without a GPU)
func TestMSLDefinesB1Kernels(t *testing.T) {
	all := allKernels()
	for _, name := range all {
		if !strings.Contains(mslSource, "kernel void "+name+"(") {
			t.Fatalf("mslSource has no kernel %q", name)
		}
	}

	if got, want := strings.Count(mslSource, "kernel void "), len(all); got != want {
		t.Fatalf("mslSource defines %d kernels, B1Kernels+B2Kernels+B3Kernels list %d", got, want)
	}
}

// Struct sizes asserted in MSL match the pkg/quant block sizes the Go side validates against
func TestMSLQuantBlockLayouts(t *testing.T) {
	for _, want := range []string{
		`static_assert(sizeof(block_q8_0) == 34`,
		`static_assert(sizeof(block_q4_0) == 18`,
		`static_assert(sizeof(block_q4_K) == 144`,
		`static_assert(sizeof(block_q5_K) == 176`,
		`static_assert(sizeof(block_q6_K) == 210`,
	} {
		if !strings.Contains(mslSource, want) {
			t.Fatalf("mslSource lacks %q", want)
		}
	}

	if quant.BlockQ8_0Size != 34 || quant.BlockQ4_0Size != 18 || quant.BlockQ4_KSize != 144 || quant.BlockQ5_KSize != 176 || quant.BlockQ6_KSize != 210 {
		t.Fatalf("pkg/quant block sizes changed: q8_0=%d q4_0=%d q4_K=%d q5_K=%d q6_K=%d", quant.BlockQ8_0Size, quant.BlockQ4_0Size, quant.BlockQ4_KSize, quant.BlockQ5_KSize, quant.BlockQ6_KSize)
	}

	// the packed ggml layouts, not the CUDA expanded 224 / 256-byte Q5_K / Q6_K blocks
	for _, bad := range []string{"== 224", "== 256"} {
		if strings.Contains(mslSource, bad) {
			t.Fatalf("mslSource asserts a CUDA expanded block size (%s)", bad)
		}
	}
}

// the Q5_K / Q6_K matvec and NORM RoPE kernels keep the CUDA-independent packed layout and the pkg/quant dequant structure
func TestMSLQ5Q6AndRoPENormKernels(t *testing.T) {
	for _, want := range []string{
		"kernel void matmul_vec_q5_k(",
		"device const block_q5_K *W [[buffer(0)]]",
		"uchar qh[32];",
		"uint u1 = 1u << (2 * g);",
		"uint u2 = 2u << (2 * g);",
		"kernel void matmul_vec_q6_k(",
		"device const block_q6_K *W [[buffer(0)]]",
		"char scales[16];",
		"half d;\n};\nstatic_assert(sizeof(block_q6_K) == 210",
		"- 32;",
		"kernel void rope_heads_norm(",
		"uint base = h * p.headDim + 2 * i;",
		"float x1 = v[base + 1];",
	} {
		if !strings.Contains(mslSource, want) {
			t.Fatalf("mslSource lacks %q", want)
		}
	}

	// every matvec kernel is 4 rows / threadgroup: one simdgroup per row
	for _, k := range []string{"matmul_vec_q5_k", "matmul_vec_q6_k"} {
		i := strings.Index(mslSource, "kernel void "+k+"(")
		body := mslSource[i : i+strings.Index(mslSource[i:], "\n}\n")]
		for _, want := range []string{"uint row = tg * ROWS_PER_TG + sg;", "simd_sum(sum)"} {
			if !strings.Contains(body, want) {
				t.Fatalf("%s lacks %q", k, want)
			}
		}
	}
}
