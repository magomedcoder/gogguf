package mempool

import "testing"

func TestAppendNAdvanceN(t *testing.T) {
	kv := NewKV(2, 8, 4, nil)
	k := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	v := []float32{
		10, 20, 30, 40,
		50, 60, 70, 80,
		90, 100, 110, 120,
	}

	kv.AppendN(0, k, v, 3)
	if got := len(kv.KLayer(0)); got != 12 {
		t.Fatalf("KLayer len=%d", got)
	}

	if kv.KLayer(0)[4] != 5 || kv.VLayer(0)[8] != 90 {
		t.Fatalf("wrong values after AppendN")
	}

	kv.AdvanceN(3)
	if kv.Len() != 3 {
		t.Fatalf("Len=%d", kv.Len())
	}
}
