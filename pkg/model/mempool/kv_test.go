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
	_ = c.Append(0, k, v)
	_ = c.Append(1, k, v)
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

	_ = c.Append(0, []float32{9, 10, 11, 12}, v)
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

func TestKVDropOldest(t *testing.T) {
	kvDim := 2
	c := NewKV(1, 4, kvDim, nil)
	for i := range 3 {
		k := []float32{float32(i), float32(i) + 0.5}
		v := []float32{float32(i) * 10, float32(i)*10 + 1}
		if err := c.Append(0, k, v); err != nil {
			t.Fatal(err)
		}
		c.Advance()
	}

	c.DropOldest(1)
	if c.Len() != 2 {
		t.Fatalf("Len=%d", c.Len())
	}

	got := c.KLayer(0)
	if len(got) != 4 || got[0] != 1 || got[2] != 2 {
		t.Fatalf("K after drop = %v", got)
	}

	c.DropOldest(10)
	if c.Len() != 0 || len(c.KLayer(0)) != 0 {
		t.Fatal("drop all")
	}
}

func TestKVEnsureRoom(t *testing.T) {
	c := NewKV(1, 3, 2, nil)
	for i := range 3 {
		if err := c.Append(0, []float32{float32(i), 0}, []float32{0, 0}); err != nil {
			t.Fatal(err)
		}

		c.Advance()
	}

	if dropped := c.EnsureRoom(1); dropped != 1 {
		t.Fatalf("EnsureRoom dropped=%d", dropped)
	}

	if c.TokensStored() != 2 || c.Len() != 2 {
		t.Fatalf("stored=%d len=%d", c.TokensStored(), c.Len())
	}

	if err := c.Append(0, []float32{9, 9}, []float32{0, 0}); err != nil {
		t.Fatal(err)
	}
}

func TestKVDropOldestLayer(t *testing.T) {
	c := NewKV(2, 4, 2, nil)
	for i := range 3 {
		k := []float32{float32(i), 0}
		_ = c.Append(0, k, k)
		_ = c.Append(1, k, k)
		c.Advance()
	}

	c.DropOldestLayer(0, 1)
	if c.LayerLen(0) != 2 || c.LayerLen(1) != 3 || c.Len() != 3 {
		t.Fatalf("layer0=%d layer1=%d len=%d", c.LayerLen(0), c.LayerLen(1), c.Len())
	}

	if got := c.KLayer(0); got[0] != 1 {
		t.Fatalf("K0=%v", got)
	}
}

func TestKVFromArena(t *testing.T) {
	kvDim := 2
	a := NewArena(2 * 2 * 4 * kvDim)
	c := NewKV(2, 4, kvDim, a)
	if a.Used() == 0 {
		t.Fatal("want Alloc from arena")
	}

	_ = c.Append(1, []float32{1, 2}, []float32{3, 4})
	if c.VLayer(1)[1] != 4 {
		t.Fatal(c.VLayer(1))
	}
}
