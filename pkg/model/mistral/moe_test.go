package mistral

import "testing"

func TestSiluDiv(t *testing.T) {
	if siluDiv(0) != 0.5 {
		t.Fatalf("siluDiv(0)=%v", siluDiv(0))
	}

	if siluDiv(1) <= 0 || siluDiv(1) >= 1 {
		t.Fatalf("siluDiv(1)=%v", siluDiv(1))
	}
}
