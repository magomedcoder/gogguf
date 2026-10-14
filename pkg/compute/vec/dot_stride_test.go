package vec

import (
	"math"
	"testing"
)

func TestDotStrideMatchesNaive(t *testing.T) {
	weights := []float32{0.1, 0.2, 0.3, 0.4}
	v := []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}
	vStride := 2

	for i := range 2 {
		got := DotStride(weights, v, i, vStride, len(weights))
		var want float32
		for t := range weights {
			want += weights[t] * v[i+t*vStride]
		}

		if math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("dim %d: got %v want %v", i, got, want)
		}
	}
}

func TestVectorMaxMatchesPure(t *testing.T) {
	x := make([]float32, 64)
	for i := range x {
		x[i] = float32(i%17) - 8
	}

	if Max(x) != maxPure(x) {
		t.Fatalf("MaxFn = %v, pure = %v", Max(x), maxPure(x))
	}
}

func TestVecScaleInPlaceMatchesPure(t *testing.T) {
	x := make([]float32, 64)
	fast := make([]float32, 64)
	for i := range x {
		x[i] = float32(i%11) - 5
		fast[i] = x[i]
	}

	scale := float32(0.25)
	scaleInPlacePure(x, scale)
	ScaleInPlace(fast, scale)

	for i := range x {
		if math.Abs(float64(x[i]-fast[i])) > 1e-6 {
			t.Fatalf("[%d] pure=%v fast=%v", i, x[i], fast[i])
		}
	}
}

func TestDotStrideMatchesPure(t *testing.T) {
	for _, n := range []int{1, 3, 7, 8, 9, 16, 31, 64, 100} {
		for _, stride := range []int{1, 2, 5, 64, 129} {
			for _, off := range []int{0, 3} {
				weights := make([]float32, n)
				for i := range weights {
					weights[i] = float32(i%13)*0.1 - 0.6
				}

				v := make([]float32, off+(n-1)*stride+1)
				for i := range v {
					v[i] = float32(i%7)*0.25 - 0.75
				}

				got := DotStride(weights, v, off, stride, n)
				want := dotStridePure(weights, v, off, stride, n)
				if math.Abs(float64(got-want)) > 1e-3*(1+math.Abs(float64(want))) {
					t.Fatalf("n=%d stride=%d off=%d: got %v want %v", n, stride, off, got, want)
				}
			}
		}
	}
}

func TestDotStrideEdgeCases(t *testing.T) {
	if got := DotStride([]float32{1, 2}, []float32{1, 2}, 0, 1, 0); got != 0 {
		t.Fatalf("n=0: got %v", got)
	}

	if got := DotStride([]float32{1, 2}, []float32{1, 2}, 0, 1, 3); got != 0 {
		t.Fatalf("n>len(weights): got %v", got)
	}
}

func benchDotStride(b *testing.B, fn func([]float32, []float32, int, int, int) float32) {
	const n, stride = 512, 128
	w := make([]float32, n)
	v := make([]float32, n*stride)
	for i := range w {
		w[i] = float32(i%5) * 0.1
	}
	for i := range v {
		v[i] = float32(i%9) * 0.1
	}

	b.ResetTimer()
	for range b.N {
		fn(w, v, 3, stride, n)
	}
}

func BenchmarkDotStride(b *testing.B) {
	benchDotStride(b, DotStride)
}

func BenchmarkDotStridePure(b *testing.B) {
	benchDotStride(b, dotStridePure)
}

func TestDotStrideBlockMatchesPure(t *testing.T) {
	for _, d := range []int{1, 3, 4, 7, 8, 12, 16, 31, 32, 40, 64, 80, 128} {
		for _, n := range []int{1, 2, 5, 33} {
			for _, extra := range []int{0, 24} {
				stride := d + extra
				off := 3
				weights := make([]float32, n)
				for i := range weights {
					weights[i] = float32(i%11)*0.1 - 0.5
				}

				v := make([]float32, off+(n-1)*stride+d)
				for i := range v {
					v[i] = float32(i%7)*0.25 - 0.75
				}

				got := make([]float32, d)
				want := make([]float32, d)
				for i := range got {
					got[i] = 99 // must be overwritten
				}

				DotStrideBlock(got, weights, v, off, stride, n)
				dotStrideBlockPure(want, weights, v, off, stride, n)
				for i := range want {
					if math.Abs(float64(got[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
						t.Fatalf("d=%d n=%d stride=%d [%d]: got %v want %v", d, n, stride, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func BenchmarkDotStrideBlock(b *testing.B) {
	const n, d, stride = 512, 128, 128
	w := make([]float32, n)
	v := make([]float32, n*stride)
	for i := range w {
		w[i] = float32(i%5) * 0.1
	}

	for i := range v {
		v[i] = float32(i%9) * 0.1
	}

	dst := make([]float32, d)
	b.ResetTimer()
	for range b.N {
		DotStrideBlock(dst, w, v, 0, stride, n)
	}
}

func BenchmarkDotStrideBlockPure(b *testing.B) {
	const n, d, stride = 512, 128, 128
	w := make([]float32, n)
	v := make([]float32, n*stride)
	dst := make([]float32, d)
	b.ResetTimer()
	for range b.N {
		dotStrideBlockPure(dst, w, v, 0, stride, n)
	}
}

func BenchmarkDotStrideLoopPerDim(b *testing.B) {
	const n, d, stride = 512, 128, 128
	w := make([]float32, n)
	v := make([]float32, n*stride)
	dst := make([]float32, d)
	b.ResetTimer()
	for range b.N {
		for i := range d {
			dst[i] = DotStride(w, v, i, stride, n)
		}
	}
}
