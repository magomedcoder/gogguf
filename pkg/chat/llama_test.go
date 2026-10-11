package chat

import (
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestFormatLlama3User(t *testing.T) {
	meta := llama3TestMeta()

	got, err := FormatUser("Hello", Options{Metadata: meta})
	if err != nil {
		t.Fatal(err)
	}

	bos := tokenFromVocab(meta, 128000)
	startHeader := tokenFromVocab(meta, llamaStartHeaderID)
	endHeader := tokenFromVocab(meta, llamaEndHeaderID)

	if !strings.HasPrefix(got, bos) {
		t.Fatalf("expected BOS at start, got %q", got[:min(32, len(got))])
	}

	userBlock := startHeader + "user" + endHeader + "\n\nHello"
	if !strings.Contains(got, userBlock) {
		t.Fatalf("expected user block %q in %q", userBlock, got)
	}

	assistantSuffix := startHeader + "assistant" + endHeader
	if !strings.HasSuffix(strings.TrimSpace(got), assistantSuffix) {
		t.Fatalf("expected suffix %q, got %q", assistantSuffix, got[len(got)-min(80, len(got)):])
	}
}

func TestFormatLlama2User(t *testing.T) {
	meta := format.Metadata{
		"general.architecture": "llama",
		"llama.rope.freq_base": float32(10000),
		format.MetaTokBOS:      int32(1),
		format.MetaTokEOS:      int32(2),
		format.MetaTokTokens:   []string{"<unk>", "<s>", "</s>", "▁Hello"},
	}

	got, err := FormatUser("Hi", Options{Metadata: meta, System: "Be nice"})
	if err != nil {
		t.Fatal(err)
	}

	if !strings.HasPrefix(got, "<s>[INST] ") {
		t.Fatalf("expected Llama2 [INST], got %q", got[:min(40, len(got))])
	}

	if !strings.Contains(got, "<<SYS>>\nBe nice\n<</SYS>>") {
		t.Fatalf("expected <<SYS>>, got %q", got)
	}

	if !strings.Contains(got, "Hi [/INST]") {
		t.Fatalf("expected user+/INST, got %q", got)
	}

	if isLlama3Family(meta) {
		t.Fatal("Llama2 meta must not be treated as Llama3")
	}
}

func llama3TestMeta() format.Metadata {
	return format.Metadata{
		"general.architecture": "llama",
		format.MetaTokBOS:      int32(128000),
		format.MetaTokEOS:      int32(128009),
		format.MetaTokPre:      "llama-bpe",
		format.MetaTokTokens:   llama3TestTokens(),
	}
}

func llama3TestTokens() []string {
	tokens := make([]string, 128010)
	tokens[128000] = "<|begin_of_text|>"
	tokens[128006] = "<|start_header_id|>"
	tokens[128007] = "<|end_header_id|>"
	tokens[128009] = "<|eot_id|>"
	return tokens
}
