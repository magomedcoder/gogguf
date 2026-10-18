//go:build !(darwin && metal)

package model

import (
	"strings"
	"testing"
)

// Without darwin+metal the macOS offload hook keeps the old hard reject and says how to enable the wiring
func TestOpenMetalGPURejectedWithoutMetalBuild(t *testing.T) {
	g, err := openMetalGPU()
	if err == nil || g != nil {
		t.Fatalf("openMetalGPU() = %v, %v; expected error", g, err)
	}

	if !strings.Contains(err.Error(), "-tags metal") {
		t.Fatalf("error %q should mention -tags metal", err)
	}
}
