package gogguf

import (
	"io"

	"github.com/magomedcoder/gogguf/pkg/chat"
	"github.com/magomedcoder/gogguf/pkg/device"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/runtime"
	"github.com/magomedcoder/gogguf/pkg/runtime/sampler"
	"github.com/magomedcoder/gogguf/pkg/tokenizer"
)

// GGUF parsing (pkg/format) and tensor dtype (pkg/dtype)

type (
	Reader       = format.Reader
	MappedReader = format.MappedReader
	Metadata     = format.Metadata
	TensorInfo   = format.TensorInfo
	Type         = format.Type // metadata value type (not tensor dtype)
	DType        = dtype.DType
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

// Inference (pkg/runtime, pkg/runtime/sampler)

type (
	Engine            = runtime.Engine
	Context           = runtime.Context
	Conversation      = runtime.Conversation
	GenerationSession = runtime.GenerationSession
	GenerateParams    = runtime.GenerateParams
	LoadOptions       = runtime.Options
	SamplerFunc       = sampler.Func
	SamplerConfig     = sampler.Config
	Tokenizer         = tokenizer.Tokenizer
	Backend           = device.Backend
)

// Load loads the model with copy-on-read weights (no mmap). Prefer LoadMapped.
// Call Engine.Close when done.
func Load(path string, opts LoadOptions) (*Engine, error) {
	return runtime.Load(path, opts)
}

// LoadMapped loads via mmap (zero-copy on Unix; full-file read on Windows).
// The Engine owns the mapping/file - call Close when done.
func LoadMapped(path string, opts LoadOptions) (*Engine, error) {
	return runtime.LoadMapped(path, opts)
}

// LoadReader builds an Engine from an open GGUF reader (tests, custom I/O).
// source is closed by Engine.Close (may be nil).
func LoadReader(r *Reader, opts LoadOptions, source io.Closer) (*Engine, error) {
	return runtime.LoadReader(r, opts, source)
}

// ParseGPUDevices parses device list ("1" / "0,1") and tensor-split ("0.6,0.4") into typed fields for LoadOptions.GPUDevices / TensorSplit (CLI helper).
func ParseGPUDevices(devices, tensorSplit string) ([]int, []float64, error) {
	return device.ParseDevices(devices, tensorSplit)
}

// NewSampler returns a next-token selection function (greedy, temperature, top-k, top-p).
func NewSampler(cfg SamplerConfig) SamplerFunc {
	return sampler.New(cfg)
}

// Greedy selects the token with the maximum logit.
var Greedy = sampler.Greedy

// Chat (pkg/chat) - templates and tools; no HTTP dependency.

type (
	ChatOptions  = chat.Options
	ChatMessage  = chat.Message
	Tool         = chat.Tool
	ToolFunction = chat.ToolFunction
	ToolCall     = chat.ToolCall
)

// FormatChatUser wraps a single user turn in a chat template.
func FormatChatUser(user string, opts ChatOptions) (string, error) {
	return chat.FormatUser(user, opts)
}

// FormatChatMessages renders a multi-turn message list (library multi-turn path).
func FormatChatMessages(messages []ChatMessage, opts ChatOptions) (string, error) {
	return chat.FormatMessages(messages, opts)
}

// HasChatTemplate reports whether tokenizer.chat_template is present in GGUF.
func HasChatTemplate(r *Reader) bool {
	return chat.HasTemplate(r)
}
