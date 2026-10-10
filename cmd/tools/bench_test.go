package main

import (
	"math"
	"testing"
)

func TestAverageBenchEmpty(t *testing.T) {
	if got := averageBench(nil); got.PrefillMS != 0 {
		t.Fatalf("expected zero result, got prefill_ms=%v", got.PrefillMS)
	}
}

func TestAverageBenchSingle(t *testing.T) {
	in := benchResult{PromptTokens: 10, PrefillMS: 100, DecodeTPS: 5}
	if got := averageBench([]benchResult{in}); got != in {
		t.Fatalf("single run should be returned unchanged: %+v", got)
	}
}

func TestAverageBenchMultiple(t *testing.T) {
	results := []benchResult{
		{
			PromptTokens: 10,
			DecodeTokens: 4,
			PrefillMS:    100,
			DecodeMS:     200,
			PrefillTPS:   100,
			DecodeTPS:    20,
		},
		{
			PromptTokens: 10,
			DecodeTokens: 6,
			PrefillMS:    200,
			DecodeMS:     300,
			PrefillTPS:   50,
			DecodeTPS:    20,
		},
	}
	got := averageBench(results)

	if got.PromptTokens != 10 {
		t.Fatalf("prompt_tokens = %d, expected 10", got.PromptTokens)
	}

	if got.DecodeTokens != 5 {
		t.Fatalf("decode_tokens = %d, expected 5", got.DecodeTokens)
	}

	if math.Abs(got.PrefillMS-150) > 1e-9 {
		t.Fatalf("prefill_ms = %v, expected 150", got.PrefillMS)
	}

	if math.Abs(got.DecodeMS-250) > 1e-9 {
		t.Fatalf("decode_ms = %v, expected 250", got.DecodeMS)
	}

	if math.Abs(got.PrefillTPS-75) > 1e-9 {
		t.Fatalf("prefill_tps = %v, expected 75", got.PrefillTPS)
	}
}
