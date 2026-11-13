//go:build metal

package metal

import (
	"strings"
	"testing"
)

// Every B1 kernel is defined in the embedded MSL (static check, runs without a GPU)
func TestMSLDefinesB1Kernels(t *testing.T) {
	for _, name := range B1Kernels {
		if !strings.Contains(mslSource, "kernel void "+name+"(") {
			t.Fatalf("mslSource has no kernel %q", name)
		}
	}

	if got, want := strings.Count(mslSource, "kernel void "), len(B1Kernels); got != want {
		t.Fatalf("mslSource defines %d kernels, B1Kernels lists %d", got, want)
	}
}
