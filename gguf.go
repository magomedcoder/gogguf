package gguf

import (
	"io"

	"github.com/magomedcoder/gguf.go/pkg/chat"
	"github.com/magomedcoder/gguf.go/pkg/format"
	"github.com/magomedcoder/gguf.go/pkg/runtime"
	"github.com/magomedcoder/gguf.go/pkg/sampler"
)

// Парсинг GGUF (pkg/format)

type (
	Reader       = format.Reader
	MappedReader = format.MappedReader
	Metadata     = format.Metadata
	TensorInfo   = format.TensorInfo
	Type         = format.Type
	GGML         = format.GGML
	Filetype     = format.Filetype
)

// OpenFile открывает GGUF-файл по пути на диске
func OpenFile(filename string) (*Reader, error) {
	return format.OpenFile(filename)
}

// OpenFileMapped открывает GGUF через memory-map (zero-copy доступ к весам)
func OpenFileMapped(filename string) (*MappedReader, error) {
	return format.OpenFileMapped(filename)
}

// Open парсит GGUF из потока для чтения тензоров источник должен реализовать io.ReaderAt
func Open(readSeeker io.ReadSeeker) (*Reader, error) {
	return format.Open(readSeeker)
}

// Inference (pkg/runtime, pkg/sampler)

type (
	Engine            = runtime.Engine
	Context           = runtime.Context
	GenerationSession = runtime.GenerationSession
	GenerateParams    = runtime.GenerateParams
	LoadOptions       = runtime.Options
	SamplerFunc       = sampler.Func
	SamplerConfig     = sampler.Config
)

// Load загружает модель и tokenizer из GGUF-файла
func Load(path string, opts LoadOptions) (*Engine, error) {
	return runtime.Load(path, opts)
}

// LoadMapped загружает модель через mmap (zero-copy веса)
func LoadMapped(path string, opts LoadOptions) (*Engine, error) {
	return runtime.LoadMapped(path, opts)
}

// NewSampler возвращает функцию выбора следующего токена (greedy, temperature, top-k, top-p)
func NewSampler(cfg SamplerConfig) SamplerFunc {
	return sampler.New(cfg)
}

// Chat (pkg/chat)

type ChatOptions = chat.Options

// FormatChatUser оборачивает промпт в chat template (Jinja из метаданных или fallback)
func FormatChatUser(user string, opts ChatOptions) (string, error) {
	return chat.FormatUser(user, opts)
}

// HasChatTemplate проверяет наличие tokenizer.chat_template в GGUF
func HasChatTemplate(r *Reader) bool {
	return chat.HasTemplate(r)
}

// Greedy выбирает токен с максимальным logit
var Greedy = sampler.Greedy
