//go:build cuda

package cuda

import (
	"errors"
	"testing"
)

// Peer API guards must not need a GPU (zero-value backends)
func TestPeerAccessRejectsInvalidPairs(t *testing.T) {
	a := &Backend{device: 0, name: "CUDA:0"}
	same := &Backend{device: 0, name: "CUDA:0 again"}
	b := &Backend{device: 1, name: "CUDA:1"}

	if err := a.EnablePeerAccess(a); !errors.Is(err, ErrPeerUnavailable) {
		t.Fatalf("self peer = %v, expected ErrPeerUnavailable", err)
	}

	if err := a.EnablePeerAccess(same); !errors.Is(err, ErrPeerUnavailable) {
		t.Fatalf("same ordinal = %v, expected ErrPeerUnavailable", err)
	}

	if err := a.EnablePeerAccess(nil); !errors.Is(err, ErrPeerUnavailable) {
		t.Fatalf("nil peer = %v, expected ErrPeerUnavailable", err)
	}

	// closed (nil ctx) devices never report peer access
	if a.PeerAccess(b) {
		t.Fatal("PeerAccess on closed devices = true")
	}

	if err := a.PeerCopyResidTo(b, 0); err == nil {
		t.Fatal("PeerCopyResidTo(n=0): expected error")
	}

	// no resident hidden on source: copy refuses before touching the driver
	if err := a.PeerCopyResidTo(b, 8); err == nil {
		t.Fatal("PeerCopyResidTo without residency: expected error")
	}
}

func TestPeerDisabledByEnv(t *testing.T) {
	t.Setenv("GOGGUF_CUDA_P2P", "0")

	a := &Backend{device: 0, name: "CUDA:0", hiddenResident: true}
	b := &Backend{device: 1, name: "CUDA:1"}

	if err := a.PeerCopyResidTo(b, 8); !errors.Is(err, ErrPeerUnavailable) {
		t.Fatalf("PeerCopyResidTo with GOGGUF_CUDA_P2P=0 = %v, expected ErrPeerUnavailable", err)
	}

	if a.PeerAccess(b) {
		t.Fatal("PeerAccess with GOGGUF_CUDA_P2P=0 = true")
	}
}
