package gpu

import "testing"

func TestCapMaxSeq(t *testing.T) {
	if got := CapMaxSeq(40960, 0); got != 4096 {
		t.Fatalf("авто-cap: получили %d, ожидали 4096", got)
	}

	if got := CapMaxSeq(2048, 0); got != 2048 {
		t.Fatalf("короткий контекст: получили %d, ожидали 2048", got)
	}

	if got := CapMaxSeq(40960, 1024); got != 1024 {
		t.Fatalf("явный лимит: получили %d, ожидали 1024", got)
	}
}
