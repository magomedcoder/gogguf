package tokenizer

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestDecodeGPT2Space(t *testing.T) {
	tok := &Tokenizer{
		tokens:     []string{"Ġhello", "Ġworld"},
		byteEncode: false,
	}

	if got := tok.Decode([]int{0, 1}); got != " hello world" {
		t.Fatalf("Decode = %q", got)
	}
}

func TestFromGGUFMetadata(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"tokenizer.ggml.model":        "gpt2",
			"tokenizer.ggml.pre":          "qwen2",
			"tokenizer.ggml.tokens":       []string{"<s>", "</s>", "Ġhi"},
			"tokenizer.ggml.merges":       []string{"Ġ h", "h i"},
			"tokenizer.ggml.bos_token_id": int32(0),
			"tokenizer.ggml.eos_token_id": int32(1),
		},
	}

	tok, err := FromGGUF(r)
	if err != nil {
		t.Fatalf("FromGGUF: %v", err)
	}

	if tok.BOS() != 0 || tok.EOS() != 1 {
		t.Fatalf("special tokens: bos=%d eos=%d", tok.BOS(), tok.EOS())
	}
	if tok.pretokenize == nil {
		t.Fatal("pretokenize not set")
	}
}
