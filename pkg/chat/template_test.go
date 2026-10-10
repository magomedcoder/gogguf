package chat

import (
	"strings"
	"testing"
)

func TestFormatUserThinkingEnabled(t *testing.T) {
	on := true
	got := formatUserFallback("Hello", Options{Thinking: &on})
	if !strings.HasSuffix(got, imStart+"assistant\n") {
		t.Fatalf("thinking enabled: expected assistant prompt without empty block, got %q", got)
	}

	open, _ := ThinkingTags(nil)
	if strings.Contains(got, open) {
		t.Fatalf("thinking enabled: unexpected thinking block in %q", got)
	}
}

func TestFormatUserThinkingDisabled(t *testing.T) {
	got := formatUserFallback("Hello", Options{})
	wantSuffix := imStart + "assistant\n" + EmptyThinkingBlock(nil)
	if !strings.HasSuffix(got, wantSuffix) {
		t.Fatalf("thinking disabled:\n got %q\nexpected suffix %q", got, wantSuffix)
	}
}

func TestApplyThinkingMode(t *testing.T) {
	base := imStart + "user\nhi" + imEnd + "\n" + imStart + "assistant\n"
	block := EmptyThinkingBlock(nil)

	off := applyThinkingMode(base, false, nil)
	if !strings.HasSuffix(off, block) {
		t.Fatalf("disabling thinking: got %q", off)
	}

	on := applyThinkingMode(base, true, nil)
	if on != base {
		t.Fatalf("enabling thinking: got %q, expected %q", on, base)
	}
}

func TestThinkingEnabledDefault(t *testing.T) {
	if ThinkingEnabled(Options{}) {
		t.Fatal("nil Thinking should default to false")
	}

	on := true
	if !ThinkingEnabled(Options{Thinking: &on}) {
		t.Fatal("explicit true should enable thinking")
	}
}

func TestEmptyThinkingBlockUsesVocabTags(t *testing.T) {
	block := EmptyThinkingBlock(nil)
	open, close := ThinkingTags(nil)
	if !strings.Contains(block, open) || !strings.Contains(block, close) {
		t.Fatalf("block %q must contain opening %q and closing %q tags", block, open, close)
	}
}
