package ops

// vecMulInPlace: a[i] *= b[i]
var vecMulInPlace = vecMulInPlacePure

func vecMulInPlacePure(a, b []float32) {
	for i := range a {
		a[i] *= b[i]
	}
}

// addInPlace: a[i] += b[i]
var addInPlace = addInPlacePure

func addInPlacePure(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// vectorMax возвращает max(x)
var vectorMax = vectorMaxPure

func vectorMaxPure(x []float32) float32 {
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

// vecScaleInPlace: x[i] *= scale
var vecScaleInPlace = vecScaleInPlacePure

func vecScaleInPlacePure(x []float32, scale float32) {
	for i := range x {
		x[i] *= scale
	}
}
