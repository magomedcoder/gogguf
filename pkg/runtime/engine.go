package runtime

import (
	"github.com/magomedcoder/gguf.go/pkg/format"
	"github.com/magomedcoder/gguf.go/pkg/model"
	"github.com/magomedcoder/gguf.go/pkg/tokenizer"
)

// Engine загружает GGUF-модель для inference
type Engine struct {
	Model model.Model
	tok   *tokenizer.Tokenizer
	meta  format.Metadata
	opts  Options
}

// LoadMapped загружает модель через mmap (zero-copy веса)
func LoadMapped(path string, opts Options) (*Engine, error) {
	mr, err := format.OpenFileMapped(path)
	if err != nil {
		return nil, err
	}
	return loadFromReader(mr.Reader, opts)
}

// Load открывает GGUF-файл и загружает модель
func Load(path string, opts Options) (*Engine, error) {
	r, err := format.OpenFile(path)
	if err != nil {
		return nil, err
	}
	return loadFromReader(r, opts)
}

func loadFromReader(r *format.Reader, opts Options) (*Engine, error) {
	m, err := model.Load(r, opts.modelOpts())
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
	}, nil
}

func (e *Engine) LoadOptions() Options {
	return e.opts
}

// Metadata возвращает KV-метаданные модели
func (e *Engine) Metadata() format.Metadata {
	return e.meta
}

// Tokenizer возвращает tokenizer модели
func (e *Engine) Tokenizer() *tokenizer.Tokenizer {
	return e.tok
}

// ContextLength возвращает максимальную длину контекста из метаданных (0 если неизвестно)
func (e *Engine) ContextLength() int {
	arch, err := e.meta.String("general.architecture")
	if err != nil || arch == "" {
		return 0
	}

	return e.meta.IntOptional(arch+".context_length", 0)
}

// ForwardTokenIDs выполняет forward pass для token IDs
func (e *Engine) ForwardTokenIDs(tokens []int, startPos int) ([]float32, error) {
	return e.Model.Forward(tokens, startPos)
}
