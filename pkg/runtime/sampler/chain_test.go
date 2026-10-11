package sampler

import (
	"testing"
)

func TestGreedy(t *testing.T) {
	logits := []float32{0.1, 0.9, 0.3}
	if got := Greedy(logits); got != 1 {
		t.Fatalf("Greedy = %d, want 1", got)
	}
}

func TestNewDeterministicWithSeed(t *testing.T) {
	logits := []float32{1, 2, 3, 4}

	s1 := New(Config{
		Temp: 0.8,
		TopK: 2,
		TopP: 0.9,
		Seed: 42,
	})
	s2 := New(Config{
		Temp: 0.8,
		TopK: 2,
		TopP: 0.9,
		Seed: 42,
	})

	a := s1(logits)
	b := s2(logits)
	if a != b {
		t.Fatalf("same seed: %d vs %d", a, b)
	}
}

func TestTopKLimitsChoice(t *testing.T) {
	logits := []float32{10, 9, 1, 0, -1}

	s := New(Config{Temp: 1, TopK: 2, Seed: 1})
	for range 50 {
		got := s(logits)
		if got > 1 {
			t.Fatalf("top-k=2 picked index %d", got)
		}
	}
}

func TestConfigGreedyWhenTempZero(t *testing.T) {
	logits := []float32{0.1, 0.5, 0.2}
	s := New(Config{Temp: 0})
	if got := s(logits); got != 1 {
		t.Fatalf("greedy config = %d", got)
	}
}

func TestApplyTopP(t *testing.T) {
	probs := []float64{0.4, 0.4, 0.1, 0.1}
	mask := []bool{true, true, true, true}
	applyTopP(probs, mask, 0.79)

	if !mask[0] || !mask[1] || mask[2] || mask[3] {
		t.Fatalf("top-p mask = %v", mask)
	}
}
