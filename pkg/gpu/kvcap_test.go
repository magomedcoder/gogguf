package gpu

import "testing"

func TestCapMaxSeq(t *testing.T) {
	if got := CapMaxSeq(40960, 0); got != 4096 {
		t.Fatalf("auto-cap: got %d, expected 4096", got)
	}

	if got := CapMaxSeq(2048, 0); got != 2048 {
		t.Fatalf("short context: got %d, expected 2048", got)
	}

	if got := CapMaxSeq(40960, 1024); got != 1024 {
		t.Fatalf("explicit limit: got %d, expected 1024", got)
	}
}
