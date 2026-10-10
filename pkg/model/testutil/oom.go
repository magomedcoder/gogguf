package testutil

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/gpu"
)

// SkipOOM skips the test when err is GPU/host OOM (shared by residency tests).
func SkipOOM(t *testing.T, err error) {
	t.Helper()
	if err != nil && gpu.IsOutOfMemory(err) {
		t.Skip("insufficient VRAM:", err)
	}
}
