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
