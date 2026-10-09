package runtime

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/model"
	"github.com/magomedcoder/gogguf/pkg/tokenizer"
)

// Engine loads a GGUF model for inference.
type Engine struct {
	Model model.Model
	tok   *tokenizer.Tokenizer
	meta  format.Metadata
	opts  Options
	gpu   gpu.Backend
}

// LoadMapped loads the model via mmap (zero-copy weights).
func LoadMapped(path string, opts Options) (*Engine, error) {
	mr, err := format.OpenFileMapped(path)
	if err != nil {
		return nil, err
	}
	return loadFromReader(mr.Reader, opts)
}

// Load opens a GGUF file and loads the model.
func Load(path string, opts Options) (*Engine, error) {
	r, err := format.OpenFile(path)
	if err != nil {
		return nil, err
	}
	return loadFromReader(r, opts)
}

func loadFromReader(r *format.Reader, opts Options) (*Engine, error) {
	// Open backend here so Engine knows the device (name, VRAM, layer plan); model.Load calls Normalize again but with GPU ready.
	mopts := opts.modelOpts()
	if err := mopts.Normalize(); err != nil {
		return nil, err
	}

	m, err := model.Load(r, mopts)
	if err != nil {
		return nil, err
	}

	tok, err := tokenizer.FromGGUF(r)
	if err != nil {
		return nil, err
	}

	return &Engine{
		Model: m,
		tok:   tok,
		meta:  r.Metadata,
		opts:  opts,
		gpu:   mopts.GPU,
	}, nil
}

// GPUBackend returns the open GPU backend (nil when -ngl 0 or CPU build).
func (e *Engine) GPUBackend() gpu.Backend {
	if e == nil {
		return nil
	}

	return e.gpu
}

// GPUDescription describes the offload device: GPU name or layer plan for multi-GPU.
// Empty string means the model is treated as CPU-only.
func (e *Engine) GPUDescription() string {
	if e == nil || e.gpu == nil {
		return ""
	}

	return gpu.Describe(e.gpu)
}

// VRAMInfo returns used/total VRAM of the backend (0, 0 without GPU).
func (e *Engine) VRAMInfo() (used, total uint64, err error) {
	if e == nil || e.gpu == nil {
		return 0, 0, nil
	}

	return e.gpu.VRAMInfo()
}

func (e *Engine) LoadOptions() Options {
	return e.opts
}

// Metadata returns model KV metadata.
func (e *Engine) Metadata() format.Metadata {
	return e.meta
}

// Tokenizer returns the model tokenizer.
func (e *Engine) Tokenizer() *tokenizer.Tokenizer {
	return e.tok
}

// ContextLength returns max context length from metadata (0 if unknown).
func (e *Engine) ContextLength() int {
	arch, err := e.meta.String("general.architecture")
	if err != nil || arch == "" {
		return 0
	}

	return e.meta.IntOptional(arch+".context_length", 0)
}

// ForwardTokenIDs runs a forward pass for token IDs.
func (e *Engine) ForwardTokenIDs(tokens []int, startPos int) ([]float32, error) {
	return e.Model.Forward(tokens, startPos)
}

// EmbedTokens returns last-token hidden embedding (output-norm); resets model KV-cache.
func (e *Engine) EmbedTokens(tokens []int) ([]float32, error) {
	if e == nil || e.Model == nil {
		return nil, fmt.Errorf("runtime: модель не загружена")
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("runtime: пустой ввод для embeddings")
	}

	return e.Model.Embed(tokens)
}

// EmbedText encodes text (with BOS when needed) and returns embedding + token IDs.
func (e *Engine) EmbedText(text string) (vec []float32, tokens []int, err error) {
	if e == nil || e.tok == nil {
		return nil, nil, fmt.Errorf("runtime: tokenizer не загружен")
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

// Close releases GPU and other model resources.
func (e *Engine) Close() error {
	if e == nil || e.Model == nil {
		return nil
	}
	err := e.Model.Close()
	e.Model = nil
	e.gpu = nil

	return err
}
