package debug

import (
	"testing"
)

func TestTopLogits(t *testing.T) {
	logits := []float32{1, 5, 3, 2, 4}
	top := TopLogits(logits, 3)

	if len(top) != 3 {
		t.Fatalf("len = %d, want 3", len(top))
	}

	want := []Logit{
		{
			ID:    1,
			Logit: 5,
		},
		{
			ID:    4,
			Logit: 4,
		},
		{
			ID:    2,
			Logit: 3,
		},
	}

	for i := range want {
		if top[i] != want[i] {
			t.Fatalf("top[%d] = %+v, want %+v", i, top[i], want[i])
		}
	}
}
