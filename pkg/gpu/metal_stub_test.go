//go:build !metal

package gpu

import (
	"errors"
	"testing"
)

// Без тега metal OpenMetal всегда отказывает и не мешает CUDA-пути
func TestOpenMetalStub(t *testing.T) {
	if _, err := OpenMetal(); !errors.Is(err, ErrMetalUnavailable) {
		t.Fatalf("OpenMetal() = %v, ожидали ErrMetalUnavailable", err)
	}
}
