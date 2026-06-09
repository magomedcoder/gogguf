package ops

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gguf.go/quant"
)

func TestMatMulVec(t *testing.T) {
	matrix := []float32{1, 2, 3, 4}
	vec := []float32{1, 0}
	out, err := MatMulVec(matrix, 2, 2, vec)
	if err != nil {
		t.Fatal(err)
	}

	if out[0] != 1 || out[1] != 3 {
		t.Fatalf("got %v", out)
	}
}

func TestRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3}
	w := []float32{1, 1, 1}
	out, err := RMSNorm(x, w, 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	var sumSq float32
	for _, v := range out {
		sumSq += v * v
	}
	
	rms := math.Sqrt(float64(sumSq / float32(len(out))))
	if math.Abs(rms-1) > 1e-4 {
		t.Fatalf("rms after norm = %v, want ~1", rms)
	}
}

func TestMatMulVecQ8_0(t *testing.T) {
	cols, rows := 32, 2
	raw := make([]byte, rows*quant.BlockQ8_0Size)
	for r := 0; r < rows; r++ {
		off := r * quant.BlockQ8_0Size
		binary.LittleEndian.PutUint16(raw[off:off+2], 0x3c00)
		for i := 0; i < quant.QK8_0; i++ {
			raw[off+2+i] = byte(int8(r + 1))
		}
	}

	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1
	}

	qOut, err := MatMulVecQ8_0(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	f32, err := quant.DequantQ8_0(raw, rows*cols)
	if err != nil {
		t.Fatal(err)
	}

	fOut, err := MatMulVec(f32, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range qOut {
		if math.Abs(float64(qOut[i]-fOut[i])) > 1e-4 {
			t.Fatalf("[%d] q8=%v f32=%v", i, qOut[i], fOut[i])
		}
	}
}
