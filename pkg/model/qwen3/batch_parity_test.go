package qwen3_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf"
)

func testModelPath(t *testing.T) string {
	t.Helper()
	candidates := []string{
		filepath.Join("..", "..", "..", "models", "Qwen3-0.6B-Q8_0.gguf"),
		filepath.Join("models", "Qwen3-0.6B-Q8_0.gguf"),
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("missing Qwen3-0.6B-Q8_0.gguf")

	return ""
}

func TestNBatchPrefillLogitsParity(t *testing.T) {
	path := testModelPath(t)

	eng1, err := gogguf.Load(path, gogguf.LoadOptions{NBatch: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer eng1.Close()

	tokens, err := eng1.Tokenizer().Encode("Hello, world!")
	if err != nil {
		t.Fatal(err)
	}

	logits1, err := eng1.ForwardTokenIDs(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}

	eng32, err := gogguf.Load(path, gogguf.LoadOptions{NBatch: 32})
	if err != nil {
		t.Fatal(err)
	}
	defer eng32.Close()

	logits32, err := eng32.ForwardTokenIDs(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}

	if len(logits1) != len(logits32) {
		t.Fatalf("len %d vs %d", len(logits1), len(logits32))
	}

	var maxAbs float64
	for i := range logits1 {
		d := math.Abs(float64(logits1[i] - logits32[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}

	if maxAbs > 1e-4 {
		t.Fatalf("max_abs logits n_batch=1 vs 32 = %g (threshold 1e-4)", maxAbs)
	}
}
