package tokenizer

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestSPMGreedyEncodeDecode(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"tokenizer.ggml.model": "llama",
			"tokenizer.ggml.tokens": []string{
				"<unk>", "<s>", "</s>", "▁Hi", "▁there", "!",
			},
			"tokenizer.ggml.bos_token_id": int32(1),
			"tokenizer.ggml.eos_token_id": int32(2),
		},
	}

	tok, err := FromGGUF(r)
	if err != nil {
		t.Fatal(err)
	}

	ids, err := tok.Encode("Hi there!")
	if err != nil {
		t.Fatal(err)
	}

	want := []int{3, 4, 5}
	if len(ids) != len(want) {
		t.Fatalf("ids=%v want %v", ids, want)
	}

	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}

	got := tok.Decode(ids)
	if got != "Hi there!" {
		t.Fatalf("Decode=%q, want %q", got, "Hi there!")
	}
}
