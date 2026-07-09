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
		"models/Qwen3-0.6B-Q8_0.gguf",
		"../models/Qwen3-0.6B-Q8_0.gguf",
		"../../models/Qwen3-0.6B-Q8_0.gguf",
	} {
		if _, err := os.Stat(p); err == nil {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}

	t.Skip("модель не найдена")

	return ""
}

func TestTokenizerHello(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	ids, err := engine.Tokenizer().Encode("Hello")
	if err != nil {
		t.Fatalf("ошибка Encode: %v", err)
	}

	if len(ids) != 1 || ids[0] != 9707 {
		t.Fatalf("Encode(Hello) = %v, ожидали [9707]", ids)
	}
}

func TestGreedyNextAfterChatPrefill(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	prompt, err := chat.FormatUser("Hello", chat.Options{
		Metadata: engine.Metadata(),
	})
	if err != nil {
		t.Fatalf("ошибка FormatUser: %v", err)
	}

	ids, err := engine.Tokenizer().Encode(prompt)
	if err != nil {
		t.Fatalf("ошибка Encode: %v", err)
	}

	engine.Model.ResetCache()
	logits, err := engine.Model.Forward(ids, 0)
	if err != nil {
		t.Fatalf("ошибка Forward: %v", err)
	}

	next := gogguf.Greedy(logits)
	// thinking выключен по умолчанию - модель начинает ответ сразу
	if next != 9707 {
		t.Fatalf("greedy next = %d, ожидали 9707 (Hello)", next)
	}
}

func TestGreedyGenerationShort(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	ctx, err := engine.NewContext()
	if err != nil {
		t.Fatalf("не удалось создать контекст: %v", err)
	}

	prompt, err := chat.FormatUser("Say hi", chat.Options{
		Metadata: engine.Metadata(),
	})
	if err != nil {
		t.Fatalf("ошибка FormatUser: %v", err)
	}

	text, err := ctx.Generate(prompt, gogguf.GenerateParams{
		MaxTokens: 4,
		Sampler:   gogguf.Greedy,
	})
	if err != nil {
		t.Fatalf("ошибка Generate: %v", err)
	}

	if text == "" {
		t.Fatal("ожидали непустую генерацию")
	}
}

func TestLoadMappedMatchesLoad(t *testing.T) {
	path := modelPath(t)

	engine, err := gogguf.Load(path, gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	mapped, err := gogguf.LoadMapped(path, gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель через mmap: %v", err)
	}

	prompt, err := chat.FormatUser("Hello", chat.Options{
		Metadata: engine.Metadata(),
	})
	if err != nil {
		t.Fatalf("ошибка FormatUser: %v", err)
	}

	ids, err := engine.Tokenizer().Encode(prompt)
	if err != nil {
		t.Fatalf("ошибка Encode: %v", err)
	}

	engine.Model.ResetCache()
	logits1, err := engine.Model.Forward(ids, 0)
	if err != nil {
		t.Fatalf("ошибка Forward (Load): %v", err)
	}

	mapped.Model.ResetCache()
	logits2, err := mapped.Model.Forward(ids, 0)
	if err != nil {
		t.Fatalf("ошибка Forward (LoadMapped): %v", err)
	}

	if gogguf.Greedy(logits1) != gogguf.Greedy(logits2) {
		t.Fatalf("Load vs LoadMapped: greedy %d != %d", gogguf.Greedy(logits1), gogguf.Greedy(logits2))
	}
}
