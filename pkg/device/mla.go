package device

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/device/mlaspec"
)

// MLAParams describes absorbed MLA (DeepSeek-V2/V3) attention sizes (see mlaspec.Params).
type MLAParams = mlaspec.Params

// MLAWeightLoader returns FP32 host weights by tensor name; only called on a device cache miss.
type MLAWeightLoader = mlaspec.WeightLoader

// MLABackend is optionally implemented by backends with device MLA attention + latent KV cache.
type MLABackend interface {
	// MLACacheInit allocates a latent KV cache for `layers` local layers (K: maxSeq x KDim, V: maxSeq x KVLora).
	MLACacheInit(layers, maxSeq int, p MLAParams) error
	// MLACacheAppend stores one token's K [KDim] and V [KVLora] for layer at pos.
	MLACacheAppend(layer, pos int, k, v []float32) error
	// MLAAttention q [NHeads][Nope+Rope] (q_nope raw, q_pe already RoPE'd) -> dst [NHeads*VHead].
	// Absorbs k_b, attends over seqLen cached positions and projects through v_b on device.
	// kbName / vbName are cached on device as FP32 (uploaded once through load).
	MLAAttention(layer int, dst, q []float32, seqLen int, p MLAParams, kbName, vbName string, load MLAWeightLoader) error
}

// AsMLABackend returns b as MLABackend when it supports device MLA.
func AsMLABackend(b Backend) (MLABackend, bool) {
	mb, ok := b.(MLABackend)
	return mb, ok
}

// MLACacheInit on MultiBackend: splits layers across devices like KVCacheInit.
func (m *MultiBackend) MLACacheInit(layers, maxSeq int, p MLAParams) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	bounds := LayerSplitBounds(layers, len(m.devs), m.split)
	for i, d := range m.devs {
		n := bounds[i+1] - bounds[i]
		if n <= 0 {
			continue
		}

		mb, ok := d.(MLABackend)
		if !ok {
			return fmt.Errorf("device: multi: %s lacks MLA support", d.Name())
		}

		if err := mb.MLACacheInit(n, maxSeq, p); err != nil {
			return fmt.Errorf("device: multi: mla init %s: %w", d.Name(), err)
		}
	}

	m.bounds = bounds
	m.layers = layers
	m.cur = m.deviceForLayer(0)

	return nil
}

// MLACacheAppend routes to the device owning layer.
func (m *MultiBackend) MLACacheAppend(layer, pos int, k, v []float32) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	mb, ok := dev.(MLABackend)
	if !ok {
		return fmt.Errorf("device: multi: %s lacks MLA support", dev.Name())
	}

	return mb.MLACacheAppend(local, pos, k, v)
}

// MLAAttention routes to the device owning layer.
func (m *MultiBackend) MLAAttention(layer int, dst, q []float32, seqLen int, p MLAParams, kbName, vbName string, load MLAWeightLoader) error {
	dev, local, err := m.route(layer)
	if err != nil {
		return err
	}

	mb, ok := dev.(MLABackend)
	if !ok {
		return fmt.Errorf("device: multi: %s lacks MLA support", dev.Name())
	}

	return mb.MLAAttention(local, dst, q, seqLen, p, kbName, vbName, load)
}
