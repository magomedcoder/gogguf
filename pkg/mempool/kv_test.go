package mempool

import "testing"

func TestArenaAlloc(t *testing.T) {
	a := NewArena(8)
	x := a.Alloc(3)
	y := a.Alloc(2)
	if len(x) != 3 || len(y) != 2 {
		t.Fatalf("len x=%d y=%d", len(x), len(y))
	}

	if a.Used() != 5 {
		t.Fatalf("Used=%d", a.Used())
	}

	x[0] = 1
	if a.buf[0] != 1 {
		t.Fatal("slices must point into slab")
	}
}

func TestArenaGrow(t *testing.T) {
	a := NewArena(2)
	_ = a.Alloc(2)
	z := a.Alloc(4)
	if len(z) != 4 {
		t.Fatalf("len=%d", len(z))
	}

	if a.Cap() < 6 {
		t.Fatalf("Cap=%d", a.Cap())
	}
}

func TestKVAppendAdvance(t *testing.T) {
	kvDim := 4
	c := NewKV(2, 8, kvDim, nil)
	k := []float32{1, 2, 3, 4}
	v := []float32{5, 6, 7, 8}
	c.Append(0, k, v)
	c.Append(1, k, v)
	if c.Len() != 0 {
		t.Fatalf("Len before Advance = %d", c.Len())
	}

	if len(c.KLayer(0)) != kvDim {
		t.Fatalf("len K=%d", len(c.KLayer(0)))
	}

	c.Advance()
	if c.Len() != 1 {
		t.Fatalf("Len after Advance = %d", c.Len())
	}

	if len(c.KLayer(0)) != kvDim {
		t.Fatalf("len K after Advance = %d", len(c.KLayer(0)))
	}

	c.Append(0, []float32{9, 10, 11, 12}, v)
	if len(c.KLayer(0)) != 2*kvDim {
		t.Fatalf("len K after 2 append = %d", len(c.KLayer(0)))
	}

	if c.KLayer(0)[4] != 9 {
		t.Fatalf("K=%v", c.KLayer(0))
	}

	c.Reset()
	if c.Len() != 0 || len(c.KLayer(0)) != 0 {
		t.Fatal("Reset")
	}
}

func TestKVFromArena(t *testing.T) {
	kvDim := 2
	a := NewArena(2 * 2 * 4 * kvDim)
	c := NewKV(2, 4, kvDim, a)
	if a.Used() == 0 {
		t.Fatal("want Alloc from arena")
	}

	c.Append(1, []float32{1, 2}, []float32{3, 4})
	if c.VLayer(1)[1] != 4 {
		t.Fatal(c.VLayer(1))
	}
}
