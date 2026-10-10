//go:build integration

package integration

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
)

func modelPath(t *testing.T) string {
	t.Helper()

	if p := os.Getenv("GGUF_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}

	for _, p := range []string{
		"models/Qwen3-1.7B-Q8_0.gguf",
		"../models/Qwen3-1.7B-Q8_0.gguf",
		"../../models/Qwen3-1.7B-Q8_0.gguf",
	} {
		if _, err := os.Stat(p); err == nil {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}

	t.Skip("model not found")

	return ""
}

func TestTokenizerHello(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	ids, err := engine.Tokenizer().Encode("Hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}

	if len(ids) != 1 || ids[0] != 9707 {
		t.Fatalf("Encode(Hello) = %v, expected [9707]", ids)
	}
}

func TestGreedyNextAfterChatPrefill(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	prompt, err := chat.FormatUser("Hello", chat.Options{
		Metadata: engine.Metadata(),
	})
	if err != nil {
		t.Fatalf("FormatUser error: %v", err)
	}

	ids, err := engine.Tokenizer().Encode(prompt)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}

	next := gogguf.Greedy(logits)
	// thinking is off by default - model starts the reply immediately
	if next != 9707 {
		t.Fatalf("greedy next = %d, expected 9707 (Hello)", next)
	}
}

func TestGreedyGenerationShort(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	ctx, err := engine.NewContext()
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}

	prompt, err := chat.FormatUser("Say hi", chat.Options{
		Metadata: engine.Metadata(),
	})
	if err != nil {
		t.Fatalf("FormatUser error: %v", err)
	}

	text, err := ctx.Generate(prompt, gogguf.GenerateParams{
		MaxTokens: 4,
		Sampler:   gogguf.Greedy,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if text == "" {
		t.Fatal("expected non-empty generation")
	}
}

func TestLoadMappedMatchesLoad(t *testing.T) {
	path := modelPath(t)

	engine, err := gogguf.Load(path, gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	mapped, err := gogguf.LoadMapped(path, gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("failed to load model via mmap: %v", err)
	}

	prompt, err := chat.FormatUser("Hello", chat.Options{
		Metadata: engine.Metadata(),
	})
	if err != nil {
		t.Fatalf("FormatUser error: %v", err)
	}

	ids, err := engine.Tokenizer().Encode(prompt)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}

	engine.Model.ResetCache()
	logits1, err := engine.Model.Forward(ids, 0)
	if err != nil {
		t.Fatalf("Forward (Load) error: %v", err)
	}

	mapped.Model.ResetCache()
	logits2, err := mapped.Model.Forward(ids, 0)
	if err != nil {
		t.Fatalf("Forward (LoadMapped) error: %v", err)
	}

	if gogguf.Greedy(logits1) != gogguf.Greedy(logits2) {
		t.Fatalf("Load vs LoadMapped: greedy %d != %d", gogguf.Greedy(logits1), gogguf.Greedy(logits2))
	}
}
