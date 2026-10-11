package sampler

import "testing"

func TestApplyRepeatPenalty(t *testing.T) {
	logits := []float32{2, 1, 3}
	history := []int{2}

	ApplyRepeatPenalty(logits, history, 2, 64)

	if logits[2] >= 3 {
		t.Fatalf("repeated token 2 must be penalized, logits[2]=%v", logits[2])
	}

	if logits[0] != 2 || logits[1] != 1 {
		t.Fatalf("non-repeated tokens must not change: %v", logits)
	}
}

func TestApplyRepeatPenaltyDisabled(t *testing.T) {
	logits := []float32{2, 1, 3}
	want := []float32{2, 1, 3}
	ApplyRepeatPenalty(logits, []int{2}, 1, 64)
	for i := range want {
		if logits[i] != want[i] {
			t.Fatalf("penalty=1: logits[%d]=%v, want %v", i, logits[i], want[i])
		}
	}
}

func TestApplyMinP(t *testing.T) {
	logits := []float32{0, 0, 10}
	ApplyMinP(logits, 0.5)

	if logits[2] <= -1e9 {
		t.Fatal("token with max prob must not be filtered")
	}

	if logits[0] > -1e9 || logits[1] > -1e9 {
		t.Fatalf("weak tokens must be filtered: %v", logits)
	}
}

func TestMinPInChain(t *testing.T) {
	logits := []float32{10, 9, 0, -10}

	s := New(Config{
		Temp: 1,
		MinP: 0.5,
		Seed: 1,
	})
	for range 50 {
		got := s(logits)
		if got > 1 {
			t.Fatalf("min-p filtered weak tokens but selected index %d", got)
		}
	}
}
