package vec

// MulInPlace a[i] *= b[i]
var MulInPlaceFn = mulInPlacePure

func mulInPlacePure(a, b []float32) {
	for i := range a {
		a[i] *= b[i]
	}
}

// AddInPlaceFn a[i] += b[i]
var AddInPlaceFn = addInPlacePure

func addInPlacePure(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// MaxFn returns max(x)
var MaxFn = maxPure

func maxPure(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}

	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}

	return m
}

// ScaleInPlaceFn x[i] *= scale
var ScaleInPlaceFn = scaleInPlacePure

func scaleInPlacePure(x []float32, scale float32) {
	for i := range x {
		x[i] *= scale
	}
}

// AddInPlace adds b to a element-wise.
func AddInPlace(a, b []float32) {
	AddInPlaceFn(a, b)
}

// MulInPlace multiplies a by b element-wise.
func MulInPlace(a, b []float32) {
	MulInPlaceFn(a, b)
}

// ScaleInPlace multiplies x by scale in-place.
func ScaleInPlace(x []float32, scale float32) {
	ScaleInPlaceFn(x, scale)
}

// Max returns max(x).
func Max(x []float32) float32 {
	return MaxFn(x)
}
