//go:build metal

package metal

import (
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

// Every B1/B2 kernel is defined in the embedded MSL (static check, runs without a GPU)
func TestMSLDefinesB1Kernels(t *testing.T) {
	all := append(append([]string(nil), B1Kernels...), B2Kernels...)
	for _, name := range all {
		if !strings.Contains(mslSource, "kernel void "+name+"(") {
			t.Fatalf("mslSource has no kernel %q", name)
		}
	}

	if got, want := strings.Count(mslSource, "kernel void "), len(all); got != want {
		t.Fatalf("mslSource defines %d kernels, B1Kernels+B2Kernels list %d", got, want)
	}
}

// Struct sizes asserted in MSL match the pkg/quant block sizes the Go side validates against
func TestMSLQuantBlockLayouts(t *testing.T) {
	for _, want := range []string{
		`static_assert(sizeof(block_q8_0) == 34`,
		`static_assert(sizeof(block_q4_0) == 18`,
		`static_assert(sizeof(block_q4_K) == 144`,
	} {
		if !strings.Contains(mslSource, want) {
			t.Fatalf("mslSource lacks %q", want)
		}
	}

	if quant.BlockQ8_0Size != 34 || quant.BlockQ4_0Size != 18 || quant.BlockQ4_KSize != 144 {
		t.Fatalf("pkg/quant block sizes changed: q8_0=%d q4_0=%d q4_K=%d", quant.BlockQ8_0Size, quant.BlockQ4_0Size, quant.BlockQ4_KSize)
	}
}
