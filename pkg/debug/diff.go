package debug

import "math"

// DiffStats - статистика сравнения двух векторов logits
type DiffStats struct {
	N         int     // число элементов
	MaxAbs    float64 // max |a-b|
	MeanAbs   float64 // mean |a-b|
	RMSE      float64
	MaxAbsIdx int // индекс max abs diff
	OverTol   int // число позиций с |diff| > Tol
	Tol       float64
}

// DiffLogits сравнивает a и b поэлементно (допуск tol для OverTol)
func DiffLogits(a, b []float32, tol float64) DiffStats {
	n := min(len(b), len(a))

	st := DiffStats{
		N:         n,
		Tol:       tol,
		MaxAbsIdx: -1,
	}
	if n == 0 {
		return st
	}

	var sumAbs, sumSq float64
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		sumAbs += d
		sumSq += d * d
		if d > st.MaxAbs {
			st.MaxAbs = d
			st.MaxAbsIdx = i
		}

		if tol > 0 && d > tol {
			st.OverTol++
		}
	}

	st.MeanAbs = sumAbs / float64(n)
	st.RMSE = math.Sqrt(sumSq / float64(n))

	return st
}

// LogSoftmaxInPlace пишет log(softmax(x)) в dst (dst может быть x)
func LogSoftmaxInPlace(dst, x []float32) {
	if len(dst) < len(x) {
		panic("debug: LogSoftmaxInPlace: dst слишком короткий")
	}

	if len(x) == 0 {
		return
	}

	maxv := x[0]
	for _, v := range x[1:] {
		if v > maxv {
			maxv = v
		}
	}

	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - maxv))
		dst[i] = float32(e)
		sum += e
	}

	inv := 1.0 / sum
	for i := range x {
		dst[i] = float32(math.Log(float64(dst[i]) * inv))
	}
}

// AlignByMax сдвигает a так, чтобы a[imax] == b[imax] (инвариант softmax)
func AlignByMax(a, b []float32) []float32 {
	if len(a) == 0 || len(b) == 0 {
		return append([]float32(nil), a...)
	}

	imax := 0
	for i := 1; i < len(a) && i < len(b); i++ {
		if a[i] > a[imax] {
			imax = i
		}
	}

	delta := b[imax] - a[imax]
	out := make([]float32, len(a))
	for i, v := range a {
		out[i] = v + delta
	}

	return out
}
