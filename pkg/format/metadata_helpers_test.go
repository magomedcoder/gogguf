package format

import "testing"

func TestMetaValueOr(t *testing.T) {
	md := Metadata{
		"present": float32(2.5),
		"wrong":   "not-a-float",
	}

	if got := MetaValueOr[float32](md, "missing", 1); got != 1 {
		t.Fatalf("missing: got %v want 1", got)
	}

	if got := MetaValueOr[float32](md, "present", 1); got != 2.5 {
		t.Fatalf("present: got %v want 2.5", got)
	}

	if got := MetaValueOr[float32](md, "wrong", 1); got != 1 {
		t.Fatalf("wrong type: got %v want 1", got)
	}

	if got := MetaValueNumberOr[int](md, "present", 0); got != 2 {
		t.Fatalf("number coerce: got %v want 2", got)
	}
}

func TestPrefixedMeta(t *testing.T) {
	md := Metadata{
		"llama.block_count":          32,
		"llama.rope.freq_base":       float32(500000),
		"llama.attention.head_count": "bad",
	}

	p := PrefixedMeta{Prefix: "llama.", MD: md}

	if p.Key("block_count") != "llama.block_count" {
		t.Fatalf("Key: got %q", p.Key("block_count"))
	}

	if v, err := p.Int("block_count"); err != nil || v != 32 {
		t.Fatalf("Int block_count: v=%v err=%v", v, err)
	}

	if got := p.IntOr("missing", 7); got != 7 {
		t.Fatalf("IntOr missing: got %v want 7", got)
	}

	if _, err := p.Int("attention.head_count"); err == nil {
		t.Fatal("Int bad type: expected error")
	}

	if got := p.Float32Or("rope.freq_base", 10000); got != 500000 {
		t.Fatalf("Float32Or: got %v want 500000", got)
	}

	if got := p.Float32Or("missing.rope", 10000); got != 10000 {
		t.Fatalf("Float32Or missing: got %v want 10000", got)
	}
}
