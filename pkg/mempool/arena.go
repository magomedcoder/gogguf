package mempool

// Arena - непрерывный float32-слаб: scratch/KV без отдельных make на каждый буфер
type Arena struct {
	buf []float32
	off int
}

// NewArena выделяет slab на capacity элементов float32
func NewArena(capacity int) *Arena {
	if capacity < 0 {
		capacity = 0
	}

	return &Arena{buf: make([]float32, capacity)}
}

// Cap возвращает ёмкость slab
func (a *Arena) Cap() int {
	if a == nil {
		return 0
	}

	return len(a.buf)
}

// Used возвращает число выделенных элементов
func (a *Arena) Used() int {
	if a == nil {
		return 0
	}

	return a.off
}

// Alloc вырезает n float32 из slab.
// При нехватке места slab расширяется
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

// Reset сбрасывает указатель выделения (существующие срезы становятся недействительны)
func (a *Arena) Reset() {
	if a != nil {
		a.off = 0
	}
}
