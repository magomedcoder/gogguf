package strutil

import (
	"slices"
	"testing"
)

func TestSplitCommaTrim(t *testing.T) {
	got := SplitCommaTrim(" 0, 1,,2 ")
	want := []string{"0", "1", "2"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}

	if SplitCommaTrim("") != nil {
		t.Fatal("empty")
	}
}
