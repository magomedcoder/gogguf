package ops

import (
	"math"
	"testing"
)

func TestRMSNormScaleMulMatchesPure(t *testing.T) {
	x := make([]float32, 128)
	w := make([]float32, 128)
	dstPure := make([]float32, 128)
	dstFast := make([]float32, 128)

	for i := range x {
		x[i] = float32(i%17) - 8
		w[i] = 1 + float32(i%5)*0.01
	}

	scale := float32(0.12345)
	rmsnormScaleMulPure(dstPure, x, w, scale)
	rmsnormScaleMul(dstFast, x, w, scale)

	for i := range dstPure {
		if math.Abs(float64(dstPure[i]-dstFast[i])) > 1e-6 {
			t.Fatalf("[%d] pure=%v fast=%v", i, dstPure[i], dstFast[i])
		}
	}
}
