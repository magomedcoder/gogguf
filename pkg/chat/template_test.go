package chat

import (
	"strings"
	"testing"
)

func TestFormatUserThinkingEnabled(t *testing.T) {
	on := true
	got := formatUserFallback("Привет", Options{Thinking: &on})
	if !strings.HasSuffix(got, imStart+"assistant\n") {
		t.Fatalf("размышление включено: ожидали промпт assistant без пустого блока, получили %q", got)
	}

	open, _ := ThinkingTags(nil)
	if strings.Contains(got, open) {
		t.Fatalf("размышление включено: неожиданный thinking-блок в %q", got)
	}
}

func TestFormatUserThinkingDisabled(t *testing.T) {
	got := formatUserFallback("Привет", Options{})
	wantSuffix := imStart + "assistant\n" + EmptyThinkingBlock(nil)
	if !strings.HasSuffix(got, wantSuffix) {
		t.Fatalf("размышление выключено:\n получили %q\nожидали суффикс %q", got, wantSuffix)
	}
}

func TestApplyThinkingMode(t *testing.T) {
	base := imStart + "user\nhi" + imEnd + "\n" + imStart + "assistant\n"
	block := EmptyThinkingBlock(nil)

	off := applyThinkingMode(base, false, nil)
	if !strings.HasSuffix(off, block) {
		t.Fatalf("выключение размышления: получили %q", off)
	}

	on := applyThinkingMode(base, true, nil)
	if on != base {
		t.Fatalf("включение размышления: получили %q, ожидали %q", on, base)
	}
}

func TestThinkingEnabledDefault(t *testing.T) {
	if ThinkingEnabled(Options{}) {
		t.Fatal("nil Thinking по умолчанию должен быть false")
	}

	on := true
	if !ThinkingEnabled(Options{Thinking: &on}) {
		t.Fatal("явный true должен включать размышление")
	}
}

func TestEmptyThinkingBlockUsesVocabTags(t *testing.T) {
	block := EmptyThinkingBlock(nil)
	open, close := ThinkingTags(nil)
	if !strings.Contains(block, open) || !strings.Contains(block, close) {
		t.Fatalf("блок %q должен содержать открывающий %q и закрывающий %q теги", block, open, close)
	}
}
