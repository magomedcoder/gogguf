package mempool

// Arena - contiguous float32 slab: scratch/KV without separate make per buffer
type Arena struct {
	buf []float32
	off int
}

// NewArena allocates slab for capacity float32 elements
func NewArena(capacity int) *Arena {
	if capacity < 0 {
		capacity = 0
	}

	return &Arena{buf: make([]float32, capacity)}
}

// Cap returns slab capacity
func (a *Arena) Cap() int {
	if a == nil {
		return 0
	}

	return len(a.buf)
}

// Used returns number of allocated elements
func (a *Arena) Used() int {
	if a == nil {
		return 0
	}

	return a.off
}

// Alloc carves n float32 from slab.
// Slab grows when out of space
func (a *Arena) Alloc(n int) []float32 {
	if n <= 0 {
		return nil
	}

	if a.off+n > len(a.buf) {
		need := a.off + n
		cap := max(max(len(a.buf)*2, need), 64)

		nb := make([]float32, cap)
		copy(nb, a.buf[:a.off])
		a.buf = nb
	}

	s := a.buf[a.off : a.off+n : a.off+n]
	a.off += n

	return s
}

// Reset resets allocation pointer (existing slices become invalid)
func (a *Arena) Reset() {
	if a != nil {
		a.off = 0
	}
}
