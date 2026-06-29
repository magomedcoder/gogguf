package runtime

import (
	"testing"

	"github.com/magomedcoder/gguf.go/pkg/format"
)

func TestEngineContextLength(t *testing.T) {
	e := &Engine{
		meta: format.Metadata{
			"general.architecture": "qwen3",
			"qwen3.context_length": int32(4096),
		},
	}
	if got := e.ContextLength(); got != 4096 {
		t.Fatalf("ContextLength() = %d, ожидали 4096", got)
	}
}

func TestEngineContextLengthUnknown(t *testing.T) {
	e := &Engine{
		meta: format.Metadata{},
	}
	if got := e.ContextLength(); got != 0 {
		t.Fatalf("ContextLength() = %d, ожидали 0", got)
	}
}
