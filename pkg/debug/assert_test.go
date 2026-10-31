package debug

import "testing"

func TestAssertClose(t *testing.T) {
	if err := AssertClose[float32](1.0, 1.001, 0.01); err != nil {
		t.Fatal(err)
	}

	if err := AssertClose(1.0, 2.0, 0.01); err == nil {
		t.Fatal("expected error")
	}
}

func TestTopKIndices(t *testing.T) {
	got := TopKIndices([]float32{1, 5, 3, 4}, 2)
	if len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Fatalf("got %v", got)
	}
}
