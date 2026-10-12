package device

import "github.com/magomedcoder/gogguf/pkg/dtype"

// EmbedGatherBackend optionally gathers token_embd on device into resident hidden
type EmbedGatherBackend interface {
	TryEmbedGather(name string, raw []byte, t dtype.DType, tokenID, vocab, embd int) error
}

// TryEmbedGather calls EmbedGatherBackend when available
func TryEmbedGather(b Backend, name string, raw []byte, t dtype.DType, tokenID, vocab, embd int) error {
	if eg, ok := b.(EmbedGatherBackend); ok {
		return eg.TryEmbedGather(name, raw, t, tokenID, vocab, embd)
	}

	return ErrHiddenUnavailable
}
