//go:build !metal

package device

import (
	"errors"
	"testing"
)

// Without metal tag OpenMetal always fails and does not affect CUDA path
func TestOpenMetalStub(t *testing.T) {
	if _, err := OpenMetal(); !errors.Is(err, ErrMetalUnavailable) {
		t.Fatalf("OpenMetal() = %v, expected ErrMetalUnavailable", err)
	}
}
