package chat

import (
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestFormatLlama3User(t *testing.T) {
	meta := llamaTestMeta()

	got, err := FormatUser("Hello", Options{Metadata: meta})
	if err != nil {
		t.Fatal(err)
	}

	bos := tokenFromVocab(meta, 128000)
	startHeader := tokenFromVocab(meta, llamaStartHeaderID)
	endHeader := tokenFromVocab(meta, llamaEndHeaderID)

	if !strings.HasPrefix(got, bos) {
		t.Fatalf("ожидали BOS в начале, получили %q", got[:min(32, len(got))])
	}

	userBlock := startHeader + "user" + endHeader + "\n\nHello"
	if !strings.Contains(got, userBlock) {
		t.Fatalf("ожидали user block %q в %q", userBlock, got)
	}

	assistantSuffix := startHeader + "assistant" + endHeader
	if !strings.HasSuffix(strings.TrimSpace(got), assistantSuffix) {
		t.Fatalf("ожидали суффикс %q, получили %q", assistantSuffix, got[len(got)-min(80, len(got)):])
	}
}

func llamaTestMeta() format.Metadata {
	return format.Metadata{
		"general.architecture":        "llama",
		"tokenizer.ggml.bos_token_id": int32(128000),
		"tokenizer.ggml.eos_token_id": int32(128009),
		"tokenizer.ggml.tokens":       llamaTestTokens(),
	}
}

func llamaTestTokens() []string {
	tokens := make([]string, 128010)
	tokens[128000] = "<|begin_of_text|>"
	tokens[128006] = "<|start_header_id|>"
	tokens[128007] = "<|end_header_id|>"
	tokens[128009] = "<|eot_id|>"
	return tokens
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
