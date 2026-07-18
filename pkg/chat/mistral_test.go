package chat

import (
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestFormatMistralInstruct(t *testing.T) {
	meta := format.Metadata{
		"general.architecture":        "mistral",
		"tokenizer.ggml.bos_token_id": int32(1),
		"tokenizer.ggml.eos_token_id": int32(2),
		"tokenizer.ggml.tokens": []string{
			"<unk>", "<s>", "</s>",
		},
	}

	got := formatMistralInstruct([]Message{
		{
			Role:    "user",
			Content: "Hello",
		},
	}, Options{
		Metadata: meta,
	})

	want := "<s>[INST] Hello [/INST]"
	if got != want {
		t.Fatalf("prompt = %q, ожидали %q", got, want)
	}

	gotSystem := formatMistralInstruct([]Message{
		{
			Role:    "system",
			Content: "Будьте кратки.",
		},
		{
			Role:    "user",
			Content: "Привет",
		},
	}, Options{
		Metadata: meta,
	})

	if !strings.Contains(gotSystem, "[INST] Будьте кратки.") {
		t.Fatalf("system отсутствует в prompt: %q", gotSystem)
	}

	if !strings.HasSuffix(gotSystem, "Привет [/INST]") {
		t.Fatalf("неожиданный суффикс: %q", gotSystem)
	}
}
