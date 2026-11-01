package prefill

import (
	"errors"
	"fmt"
	"testing"
)

func TestRunEmptyBatch(t *testing.T) {
	err := Run("llama", 8, 1, 0, nil, false, nil, nil, nil, nil, nil)
	if err == nil || err.Error() != "llama: empty batch" {
		t.Fatalf("got %v", err)
	}
}

func TestRunBatchTooLarge(t *testing.T) {
	err := Run("mistral", 2, 1, 0, []int{1, 2, 3}, false, nil, nil, nil, nil, nil)
	if err == nil || err.Error() != "mistral: batch=3 > n_batch=2" {
		t.Fatalf("got %v", err)
	}
}

func TestRunOrder(t *testing.T) {
	var order []string
	var seenStart int
	err := Run("test", 4, 2, 7, []int{1, 2}, true,
		func() error { order = append(order, "sync"); return nil },
		func(batch int) error {
			if batch != 2 {
				t.Fatalf("embed batch=%d", batch)
			}
			order = append(order, "embed")
			return nil
		},
		func(layer, startPos, batch int) error {
			seenStart = startPos
			order = append(order, fmt.Sprintf("layer%d", layer))
			return nil
		},
		func(batch int) {
			order = append(order, "advance")
		},
		func(batch int) error {
			order = append(order, "logits")
			return nil
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	if seenStart != 7 {
		t.Fatalf("startPos=%d, want 7", seenStart)
	}

	want := []string{"sync", "embed", "layer0", "layer1", "advance", "logits"}
	if len(order) != len(want) {
		t.Fatalf("order=%v want=%v", order, want)
	}

	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order=%v want=%v", order, want)
		}
	}
}

func TestRunSkipsLogits(t *testing.T) {
	called := false
	err := Run("test", 4, 0, 0, []int{1}, false, nil,
		func(int) error {
			return nil
		},
		nil,
		func(int) {},
		func(int) error {
			called = true
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	if called {
		t.Fatal("logits should be skipped")
	}
}

func TestRunSyncError(t *testing.T) {
	syncErr := errors.New("sync fail")
	err := Run("qwen3", 4, 0, 0, []int{1}, false,
		func() error {
			return syncErr
		},
		nil, nil, nil, nil,
	)

	if !errors.Is(err, syncErr) {
		t.Fatalf("got %v", err)
	}
}
