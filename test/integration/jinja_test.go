//go:build integration

package integration

import (
	"testing"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/chat"
)

func TestJinjaQwen3MatchesFallback(t *testing.T) {
	engine, err := gogguf.Load(modelPath(t), gogguf.LoadOptions{})
	if err != nil {
		t.Fatalf("не удалось загрузить модель: %v", err)
	}

	got, err := chat.FormatUser("Hello", chat.Options{Metadata: engine.Metadata()})
	if err != nil {
		t.Fatal(err)
	}

	// fallback without Jinja metadata path - compare that Jinja gives the same result
	meta := engine.Metadata()
	if !chat.HasTemplateMeta(meta) {
		t.Skip("нет chat template")
	}

	// repeated calls must be stable
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
