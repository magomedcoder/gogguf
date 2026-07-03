package ops

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gguf.go/pkg/quant"
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

func TestMatMulVecParallel(t *testing.T) {
	rows, cols := 128, 8
	matrix := make([]float32, rows*cols)
	vec := make([]float32, cols)
	for i := range matrix {
		matrix[i] = float32(i%11) - 5
	}

	for i := range vec {
		vec[i] = float32(i) * 0.1
	}

	got, err := MatMulVec(matrix, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for r := range rows {
		var want float32
		for c := range cols {
			want += matrix[r*cols+c] * vec[c]
		}

		if math.Abs(float64(got[r]-want)) > 1e-5 {
			t.Fatalf("row %d: got %v want %v", r, got[r], want)
		}
	}
}

func TestVectorRMS(t *testing.T) {
	x := []float32{3, 4}
	want := float32(math.Sqrt(12.5))
	if got := VectorRMS(x); math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("VectorRMS = %v, want %v", got, want)
	}
}

func TestMatMulVecQ8_0Parallel(t *testing.T) {
	cols, rows := 32, 128
	raw := make([]byte, rows*quant.BlockQ8_0Size)
	for r := range rows {
		off := r * quant.BlockQ8_0Size
		binary.LittleEndian.PutUint16(raw[off:off+2], 0x3c00)
		for i := range quant.QK8_0 {
			raw[off+2+i] = byte(int8((r + i) % 7))
		}
	}

	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = float32(i)*0.1 - 0.5
	}

	got, err := MatMulVecQ8_0(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	f32, err := quant.DequantQ8_0(raw, rows*cols)
	if err != nil {
		t.Fatal(err)
	}

	want, err := MatMulVec(f32, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("[%d] parallel q8=%v f32=%v", i, got[i], want[i])
		}
	}
}

func TestMatMulVecQ8_0(t *testing.T) {
	cols, rows := 32, 2
	raw := make([]byte, rows*quant.BlockQ8_0Size)
	for r := range rows {
		off := r * quant.BlockQ8_0Size
		binary.LittleEndian.PutUint16(raw[off:off+2], 0x3c00)
		for i := range quant.QK8_0 {
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
