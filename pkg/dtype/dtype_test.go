package dtype

import "testing"

func TestStringAndDataSize(t *testing.T) {
	if Q8_0.String() != "q8_0" {
		t.Fatalf("String=%q", Q8_0.String())
	}

	if got := F32.DataSize([]uint64{2, 2}); got != 16 {
		t.Fatalf("DataSize=%d", got)
	}
}
