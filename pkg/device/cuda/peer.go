//go:build cuda

package cuda

/*
#include "driver.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"strings"
)

// ErrPeerUnavailable P2P between two CUDA devices is not possible (driver, topology or GOGGUF_CUDA_P2P=0)
var ErrPeerUnavailable = errors.New("cuda: peer access unavailable")

// peerDisabledByEnv: GOGGUF_CUDA_P2P=0/off/false forces the host bounce for hidden transfers between devices
func peerDisabledByEnv() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GOGGUF_CUDA_P2P"))) {
	case "0", "off", "false", "no":
		return true
	}

	return false
}

// peerLockPair locks two backends on distinct devices in device-ordinal order and returns the unlock func
func peerLockPair(a, b *Backend) func() {
	first, second := a, b
	if b.device < a.device {
		first, second = b, a
	}

	first.mu.Lock()
	second.mu.Lock()

	return func() {
		second.mu.Unlock()
		first.mu.Unlock()
	}
}

// PeerAccess reports whether hidden state can move to dst over P2P (peer access enabled both ways).
// The first successful call enables access; results are cached per pair.
func (b *Backend) PeerAccess(dst *Backend) bool {
	return b.EnablePeerAccess(dst) == nil
}

// EnablePeerAccess verifies that b and dst can address each other's memory and enables access in both directions.
// Returns ErrPeerUnavailable (wrapped) when P2P is impossible; callers then fall back to a host bounce.
func (b *Backend) EnablePeerAccess(dst *Backend) error {
	if b == nil || dst == nil || b == dst || b.device == dst.device {
		return fmt.Errorf("%w: need two distinct devices", ErrPeerUnavailable)
	}

	if peerDisabledByEnv() {
		return fmt.Errorf("%w: disabled by GOGGUF_CUDA_P2P", ErrPeerUnavailable)
	}

	unlock := peerLockPair(b, dst)
	defer unlock()

	return b.enablePeerLocked(dst)
}

// enablePeerLocked requires b.mu and dst.mu held
func (b *Backend) enablePeerLocked(dst *Backend) error {
	if b.ctx == nil || dst.ctx == nil {
		return fmt.Errorf("%w: %s <-> %s: device closed", ErrPeerUnavailable, b.name, dst.name)
	}

	if ok, seen := b.peers[dst]; seen {
		if ok {
			return nil
		}

		return fmt.Errorf("%w: %s <-> %s", ErrPeerUnavailable, b.name, dst.name)
	}

	if b.peers == nil {
		b.peers = make(map[*Backend]bool)
	}

	if dst.peers == nil {
		dst.peers = make(map[*Backend]bool)
	}

	fail := func(why string) error {
		b.peers[dst] = false
		dst.peers[b] = false

		return fmt.Errorf("%w: %s <-> %s: %s", ErrPeerUnavailable, b.name, dst.name, why)
	}

	if C.gguf_cuda_peer_can_access(&b.drv, &dst.drv) == 0 {
		return fail("cuDeviceCanAccessPeer reports no P2P (topology or driver)")
	}

	if rc := C.gguf_cuda_peer_enable(&b.drv, b.ctx, dst.ctx); rc != 0 {
		return fail(fmt.Sprintf("cuCtxEnablePeerAccess %s -> %s: code %d", b.name, dst.name, int(rc)))
	}

	if rc := C.gguf_cuda_peer_enable(&dst.drv, dst.ctx, b.ctx); rc != 0 {
		return fail(fmt.Sprintf("cuCtxEnablePeerAccess %s -> %s: code %d", dst.name, b.name, int(rc)))
	}

	b.peers[dst] = true
	dst.peers[b] = true

	return nil
}

// PeerCopyResidTo moves the resident hidden state (d_resid, n floats) of b to dst with cuMemcpyPeer, no host bounce.
// After success dst holds the resident hidden state and b's residency is cleared (same contract as HiddenDownload+HiddenUpload).
// On error b keeps its residency, so the caller can fall back to the host path.
func (b *Backend) PeerCopyResidTo(dst *Backend, n int) error {
	if n <= 0 {
		return fmt.Errorf("cuda: PeerCopyResidTo: n=%d", n)
	}

	if b == nil || dst == nil || b == dst || b.device == dst.device {
		return fmt.Errorf("%w: need two distinct devices", ErrPeerUnavailable)
	}

	if peerDisabledByEnv() {
		return fmt.Errorf("%w: disabled by GOGGUF_CUDA_P2P", ErrPeerUnavailable)
	}

	unlock := peerLockPair(b, dst)
	defer unlock()

	if !b.hiddenResident {
		return fmt.Errorf("cuda: PeerCopyResidTo: residency not active on %s", b.name)
	}

	if err := b.enablePeerLocked(dst); err != nil {
		return err
	}

	rc := C.gguf_cuda_hidden_copy_peer(&b.drv, b.ctx, &b.matmulPool, &dst.drv, dst.ctx, &dst.matmulPool, C.int(n))
	if rc != 0 {
		dst.clearResidencyLocked()

		return fmt.Errorf("cuda: peer copy %s -> %s: code %d", b.name, dst.name, int(rc))
	}

	b.clearResidencyLocked()

	dst.matmulPool.keep_resid_device = 1
	dst.hiddenResident = true
	dst.lastVecAddr = 0
	dst.lastVecLen = 0

	return nil
}
