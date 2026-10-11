package chat

import (
	"strings"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestFormatMistralInstruct(t *testing.T) {
	meta := format.Metadata{
		"general.architecture": "mistral",
		format.MetaTokBOS:      int32(1),
		format.MetaTokEOS:      int32(2),
		format.MetaTokTokens: []string{
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
		t.Fatalf("prompt = %q, expected %q", got, want)
	}

	gotSystem := formatMistralInstruct([]Message{
		{
			Role:    "system",
			Content: "Be brief.",
		},
		{
			Role:    "user",
			Content: "Hello",
		},
	}, Options{
		Metadata: meta,
	})

	if !strings.Contains(gotSystem, "[INST] Be brief.") {
		t.Fatalf("system missing from prompt: %q", gotSystem)
	}

	if !strings.HasSuffix(gotSystem, "Hello [/INST]") {
		t.Fatalf("unexpected suffix: %q", gotSystem)
	}
}
