package gogguf

import (
	"io"

	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/gpu"
	"github.com/magomedcoder/gogguf/pkg/runtime"
	"github.com/magomedcoder/gogguf/pkg/sampler"
)

// GGUF parsing (pkg/format)

type (
	Reader       = format.Reader
	MappedReader = format.MappedReader
	Metadata     = format.Metadata
	TensorInfo   = format.TensorInfo
	Type         = format.Type
	GGML         = format.GGML
	Filetype     = format.Filetype
)

// OpenFile opens a GGUF file at a path on disk.
func OpenFile(filename string) (*Reader, error) {
	return format.OpenFile(filename)
}

// OpenFileMapped opens GGUF via memory-map (zero-copy access to weights).
func OpenFileMapped(filename string) (*MappedReader, error) {
	return format.OpenFileMapped(filename)
}

// Open parses GGUF from a stream; the source must implement io.ReaderAt to read tensors.
func Open(readSeeker io.ReadSeeker) (*Reader, error) {
	return format.Open(readSeeker)
}

// Inference (pkg/runtime, pkg/sampler)

type (
	Engine            = runtime.Engine
	Context           = runtime.Context
	Conversation      = runtime.Conversation
	GenerationSession = runtime.GenerationSession
	GenerateParams    = runtime.GenerateParams
	LoadOptions       = runtime.Options
	SamplerFunc       = sampler.Func
	SamplerConfig     = sampler.Config
)

// Load loads the model and tokenizer from a GGUF file.
func Load(path string, opts LoadOptions) (*Engine, error) {
	return runtime.Load(path, opts)
}

// LoadMapped loads the model via mmap (zero-copy weights).
func LoadMapped(path string, opts LoadOptions) (*Engine, error) {
	return runtime.LoadMapped(path, opts)
}

// ParseGPUDevices parses -dev ("1" / "0,1") and -tensor-split ("0.6,0.4") into device list and layer splits for LoadOptions.
func ParseGPUDevices(devices, tensorSplit string) ([]int, []float64, error) {
	return gpu.ParseDevices(devices, tensorSplit)
}

// NewSampler returns a next-token selection function (greedy, temperature, top-k, top-p).
func NewSampler(cfg SamplerConfig) SamplerFunc {
	return sampler.New(cfg)
}

// Chat (pkg/chat)

type ChatOptions = chat.Options

// FormatChatUser wraps the prompt in a chat template (Jinja from metadata or fallback).
func FormatChatUser(user string, opts ChatOptions) (string, error) {
	return chat.FormatUser(user, opts)
}

// HasChatTemplate reports whether tokenizer.chat_template is present in GGUF.
func HasChatTemplate(r *Reader) bool {
	return chat.HasTemplate(r)
}

// Greedy selects the token with the maximum logit.
var Greedy = sampler.Greedy
