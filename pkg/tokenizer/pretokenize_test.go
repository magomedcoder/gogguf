package tokenizer

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestPretokenizeLlamaBPE(t *testing.T) {
	got := pretokenizeLlamaBPE("12345")
	want := []string{"123", "45"}
	if len(got) != len(want) {
		t.Fatalf("pretokenizeLlamaBPE = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pretokenizeLlamaBPE[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestFromGGUFLlamaBPE(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"tokenizer.ggml.model":  "gpt2",
			"tokenizer.ggml.pre":    "llama-bpe",
			"tokenizer.ggml.tokens": []string{"a"},
		},
	}
	tok, err := FromGGUF(r)
	if err != nil {
		t.Fatal(err)
	}
	if tok.pretokenize == nil {
		t.Fatal("pretokenize not set")
	}
}
