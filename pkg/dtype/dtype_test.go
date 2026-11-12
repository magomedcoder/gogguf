package dtype

import "testing"

func TestStringAndDataSize(t *testing.T) {
	if Q8_0.String() != "q8_0" {
		t.Fatalf("String=%q", Q8_0.String())
	}

	if got := F32.DataSize([]uint64{2, 2}); got != 16 {
		t.Fatalf("DataSize=%d", got)
	}

	if Q8_1.DataSize([]uint64{32}) != 36 {
		t.Fatalf("Q8_1 DataSize=%d", Q8_1.DataSize([]uint64{32}))
	}

	if BF16.DataSize([]uint64{4}) != 8 {
		t.Fatalf("BF16 DataSize=%d", BF16.DataSize([]uint64{4}))
	}

	if IQ2_XXS.DataSize([]uint64{256}) != 66 {
		t.Fatalf("IQ2_XXS DataSize=%d", IQ2_XXS.DataSize([]uint64{256}))
	}

	if Int8 != 24 || Int16 != 25 || Int32 != 26 {
		t.Fatalf("integer type IDs shifted: Int8=%d Int16=%d Int32=%d", Int8, Int16, Int32)
	}
}
