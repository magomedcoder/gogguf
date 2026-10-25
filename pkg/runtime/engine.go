package runtime

import (
	"fmt"
	"io"

	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/model"
	"github.com/magomedcoder/gogguf/pkg/tokenizer"
)

// Engine loads a GGUF model for inference.
//
// Concurrency: at most one Forward/Generate in flight per Engine (shared KV).
// pkg/server serializes with a mutex; library callers must do the same if sharing an Engine.
//
// Advanced: Model is the architecture implementation (import pkg/model for the interface).
// Prefer NewContext / Conversation / EmbedText for typical library use.
type Engine struct {
	Model  model.Model
	tok    *tokenizer.Tokenizer
	meta   format.Metadata
	opts   Options
	dev    device.Backend
	source io.Closer // file and/or mmap; closed by Close
}

// LoadMapped loads the model via mmap (zero-copy weights on Unix; full read on Windows).
// The Engine owns the mapping/file; call Close when done.
func LoadMapped(path string, opts Options) (*Engine, error) {
	mr, err := format.OpenFileMapped(path)
	if err != nil {
		return nil, err
	}

	eng, err := LoadReader(mr.Reader, opts, mr)
	if err != nil {
		_ = mr.Close()
		return nil, err
	}

	return eng, nil
}

// Load opens a GGUF file with copy-on-read weights (no mmap). Prefer LoadMapped.
// The Engine owns the file handle; call Close when done.
func Load(path string, opts Options) (*Engine, error) {
	r, closer, err := format.OpenFileOwned(path)
	if err != nil {
		return nil, err
	}

	eng, err := LoadReader(r, opts, closer)
	if err != nil {
		_ = closer.Close()
		return nil, err
	}

	return eng, nil
}

// LoadReader builds an Engine from an already-open GGUF reader.
// source is closed by Engine.Close (may be nil).
func LoadReader(r *format.Reader, opts Options, source io.Closer) (*Engine, error) {
	mopts := opts
	if err := mopts.Normalize(); err != nil {
		return nil, err
	}

	m, err := model.Load(r, mopts)
	if err != nil {
		return nil, err
	}

	tok, err := tokenizer.FromGGUF(r)
	if err != nil {
		_ = m.Close()
		return nil, err
	}

	return &Engine{
		Model:  m,
		tok:    tok,
		meta:   r.Metadata,
		opts:   opts,
		dev:    mopts.GPU,
		source: source,
	}, nil
}

// GPUBackend returns the open GPU backend (nil when -ngl 0 or CPU build).
func (e *Engine) GPUBackend() device.Backend {
	if e == nil {
		return nil
	}

	return e.dev
}

// GPUDescription describes the offload device: GPU name or layer plan for multi-GPU.
// Empty string means the model is treated as CPU-only.
func (e *Engine) GPUDescription() string {
	if e == nil || e.dev == nil {
		return ""
	}

	return device.Describe(e.dev)
}

// VRAMInfo returns used/total VRAM of the backend (0, 0 without GPU).
func (e *Engine) VRAMInfo() (used, total uint64, err error) {
	if e == nil || e.dev == nil {
		return 0, 0, nil
	}

	return e.dev.VRAMInfo()
}

func (e *Engine) LoadOptions() Options {
	return e.opts
}

// PrefillBatchSize returns the model's effective n_batch (1 for MoE/MLA/SWA serial).
func (e *Engine) PrefillBatchSize() int {
	if e == nil || e.Model == nil {
		return 1
	}

	type batchSizer interface{ PrefillBatchSize() int }
	if s, ok := e.Model.(batchSizer); ok {
		n := s.PrefillBatchSize()
		if n < 1 {
			return 1
		}

		return n
	}

	return 1
}

// Metadata returns model KV metadata.
func (e *Engine) Metadata() format.Metadata {
	return e.meta
}

// Tokenizer returns the model tokenizer.
func (e *Engine) Tokenizer() *tokenizer.Tokenizer {
	return e.tok
}

// Architecture returns general.architecture from metadata (empty if unknown).
func (e *Engine) Architecture() string {
	if e == nil || e.meta == nil {
		return ""
	}

	arch, err := e.meta.String("general.architecture")
	if err != nil {
		return ""
	}

	return arch
}

// ContextLength returns max context length from metadata (0 if unknown).
func (e *Engine) ContextLength() int {
	arch := e.Architecture()
	if arch == "" {
		return 0
	}

	return e.meta.IntOptional(arch+".context_length", 0)
}

// EmbeddingLength returns embedding/hidden size from metadata (0 if unknown).
func (e *Engine) EmbeddingLength() int {
	arch := e.Architecture()
	if arch == "" {
		return 0
	}

	if n := e.meta.IntOptional(arch+".embedding_length", 0); n > 0 {
		return n
	}

	return e.meta.IntOptional(arch+".hidden_size", 0)
}

// ForwardTokenIDs runs a forward pass for token IDs.
func (e *Engine) ForwardTokenIDs(tokens []int, startPos int) ([]float32, error) {
	return e.Model.Forward(tokens, startPos)
}

// EmbedTokens returns last-token hidden embedding (output-norm); resets model KV-cache.
func (e *Engine) EmbedTokens(tokens []int) ([]float32, error) {
	if e == nil || e.Model == nil {
		return nil, fmt.Errorf("runtime: model not loaded")
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("runtime: empty input for embeddings")
	}

	return e.Model.Embed(tokens)
}

// EmbedText encodes text (with BOS when needed) and returns embedding + token IDs.
func (e *Engine) EmbedText(text string) (vec []float32, tokens []int, err error) {
	if e == nil || e.tok == nil {
		return nil, nil, fmt.Errorf("runtime: tokenizer not loaded")
	}

	tokens, err = e.encodeForEmbed(text)
	if err != nil {
		return nil, nil, err
	}

	vec, err = e.EmbedTokens(tokens)
	return vec, tokens, err
}

func (e *Engine) encodeForEmbed(text string) ([]int, error) {
	ids, err := e.tok.Encode(text)
	if err != nil {
		return nil, err
	}

	if !needsBOSPrefix(e.meta, ids) {
		return ids, nil
	}

	bos := e.tok.BOS()
	if bos < 0 {
		return ids, nil
	}

	return append([]int{bos}, ids...), nil
}

// Close releases GPU, model resources, and the underlying file/mmap.
func (e *Engine) Close() error {
	if e == nil {
		return nil
	}

	var err error
	if e.Model != nil {
		err = e.Model.Close()
		e.Model = nil
	}

	e.dev = nil
	if e.source != nil {
		if e2 := e.source.Close(); err == nil {
			err = e2
		}
		e.source = nil
	}

	return err
}
