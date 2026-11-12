package device

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// FFNSwiGLUMatBackend optionally runs batch FFN with activations kept on device between gate/up/SwiGLU/down
type FFNSwiGLUMatBackend interface {
	FFNSwiGLUMatCached(gateName, upName, downName string, gateW, upW, downW, x, out []float32, embd, ffn, batch int) error

	FFNSwiGLUMatQuantCached(t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn, batch int) error
}

// FFNSwiGLUMatQuantCached dispatches batch fused FFN when the backend supports it
func FFNSwiGLUMatQuantCached(b Backend, t dtype.DType, gateName, upName, downName string, gateRaw, upRaw, downRaw []byte, x, out []float32, embd, ffn, batch int) error {
	if mb, ok := b.(FFNSwiGLUMatBackend); ok {
		return mb.FFNSwiGLUMatQuantCached(t, gateName, upName, downName, gateRaw, upRaw, downRaw, x, out, embd, ffn, batch)
	}

	return fmt.Errorf("device: FFNSwiGLUMatQuantCached unavailable")
}

// FFNSwiGLUMatCachedFP32 dispatches FP32 batch fused FFN when available
func FFNSwiGLUMatCachedFP32(b Backend, gateName, upName, downName string, gateW, upW, downW, x, out []float32, embd, ffn, batch int) error {
	if mb, ok := b.(FFNSwiGLUMatBackend); ok {
		return mb.FFNSwiGLUMatCached(gateName, upName, downName, gateW, upW, downW, x, out, embd, ffn, batch)
	}

	return fmt.Errorf("device: FFNSwiGLUMatCached unavailable")
}
