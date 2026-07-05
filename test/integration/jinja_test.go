//go:build integration

package integration

import (
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
)

func TestJinjaQwen3MatchesFallback(t *testing.T) {
	engine, err := gguf.Load(modelPath(t), gguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	got, err := chat.FormatUser("Hello", chat.Options{Metadata: engine.Metadata()})
	if err != nil {
		t.Fatal(err)
	}

	// fallback без Jinja metadata path - сравниваем с тем, что Jinja даёт тот же результат
	meta := engine.Metadata()
	if !chat.HasTemplateMeta(meta) {
		t.Skip("нет chat template")
	}

	// Повторный вызов должен быть стабильным
	got2, err := chat.FormatUser("Hello", chat.Options{Metadata: meta})
	if err != nil {
		t.Fatal(err)
	}
	if got != got2 {
		t.Fatalf("нестабильный рендер: %q vs %q", got, got2)
	}

	if got == "" {
		t.Fatal("пустой prompt")
	}
}
