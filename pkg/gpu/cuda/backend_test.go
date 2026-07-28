//go:build cuda

package cuda

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/ops"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

func TestMatMulVecGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}
	defer b.Close()

	matrix := []float32{1, 2, 3, 4}
	vec := []float32{1, 0}
	want, err := ops.MatMulVec(matrix, 2, 2, vec)
	if err != nil {
		t.Fatal(err)
	}

	got, err := b.MatMulVec(matrix, 2, 2, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("строка %d: получили %v, ожидали %v", i, got[i], want[i])
		}
	}
}

func TestMatMulVecCachedGraphReplay(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}
	defer b.Close()

	rows, cols := 4, 8
	matrix := make([]float32, rows*cols)
	for i := range matrix {
		matrix[i] = float32(i%7) + 0.5
	}

	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = float32(i+1) * 0.1
	}

	want, err := ops.MatMulVec(matrix, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	// Первый вызов: HtoD + capture full graph
	got1, err := b.MatMulVecCached("graph-fp32", matrix, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	// Второй: same vec -> kernel-only graph (без HtoD)
	got2, err := b.MatMulVecCached("graph-fp32", matrix, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got1[i]-want[i])) > 1e-4 {
			t.Fatalf("pass1 строка %d: получили %v, ожидали %v", i, got1[i], want[i])
		}

		if math.Abs(float64(got2[i]-want[i])) > 1e-4 {
			t.Fatalf("pass2 строка %d: получили %v, ожидали %v", i, got2[i], want[i])
		}
	}

	// Другой vec - staging + тот же graph
	vec2 := make([]float32, cols)
	for i := range vec2 {
		vec2[i] = float32(cols - i)
	}

	want2, err := ops.MatMulVec(matrix, rows, cols, vec2)
	if err != nil {
		t.Fatal(err)
	}

	got3, err := b.MatMulVecCached("graph-fp32", matrix, rows, cols, vec2)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want2 {
		if math.Abs(float64(got3[i]-want2[i])) > 1e-4 {
			t.Fatalf("vec2 строка %d: получили %v, ожидали %v", i, got3[i], want2[i])
		}
	}

	if !b.hasGraphs {
		t.Log("CUDA Graphs API недоступен - используется pool + обычный launch")
	}
}

// TestMatMulVecCachedKernelOnlySameVec: один h -> две матрицы (как Q/K), второй matmul без HtoD
func TestMatMulVecCachedKernelOnlySameVec(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}
	defer b.Close()

	rows, cols := 8, 16
	wq := make([]float32, rows*cols)
	wk := make([]float32, rows*cols)
	for i := range wq {
		wq[i] = float32(i%5) * 0.25
		wk[i] = float32((i*3)%7) * 0.1
	}

	h := make([]float32, cols)
	for i := range h {
		h[i] = float32(i+1) * 0.05
	}

	wantQ, err := ops.MatMulVec(wq, rows, cols, h)
	if err != nil {
		t.Fatal(err)
	}

	wantK, err := ops.MatMulVec(wk, rows, cols, h)
	if err != nil {
		t.Fatal(err)
	}

	gotQ, err := b.MatMulVecCached("ko-q", wq, rows, cols, h)
	if err != nil {
		t.Fatal(err)
	}

	gotK, err := b.MatMulVecCached("ko-k", wk, rows, cols, h)
	if err != nil {
		t.Fatal(err)
	}

	for i := range wantQ {
		if math.Abs(float64(gotQ[i]-wantQ[i])) > 1e-4 {
			t.Fatalf("Q[%d]=%v want %v", i, gotQ[i], wantQ[i])
		}

		if math.Abs(float64(gotK[i]-wantK[i])) > 1e-4 {
			t.Fatalf("K[%d]=%v want %v", i, gotK[i], wantK[i])
		}
	}

	// повтор K: снова kernel-only replay
	gotK2, err := b.MatMulVecCached("ko-k", wk, rows, cols, h)
	if err != nil {
		t.Fatal(err)
	}

	for i := range wantK {
		if math.Abs(float64(gotK2[i]-wantK[i])) > 1e-4 {
			t.Fatalf("K replay[%d]=%v want %v", i, gotK2[i], wantK[i])
		}
	}
}

func BenchmarkMatMulVecCachedGPU(b *testing.B) {
	be, err := Open()
	if err != nil {
		b.Skip("CUDA недоступна:", err)
	}
	defer be.Close()

	rows, cols := 1024, 1024
	matrix := make([]float32, rows*cols)
	vec := make([]float32, cols)
	for i := range matrix {
		matrix[i] = 0.001
	}
	for i := range vec {
		vec[i] = 1
	}

	// прогрев: upload + graph capture
	if _, err := be.MatMulVecCached("bench", matrix, rows, cols, vec); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, err := be.MatMulVecCached("bench", matrix, rows, cols, vec); err != nil {
			b.Fatal(err)
		}
	}
}

func TestMatMulVecQ8_0GPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}
	defer b.Close()

	// 2x32 Q8_0: row0 = 1..32, row1 = 32..1
	rows, cols := 2, 32
	vec := make([]float32, cols)
	for i := range cols {
		vec[i] = 1
	}

	raw := make([]byte, rows*(cols/quant.QK8_0)*quant.BlockQ8_0Size)
	for r := range rows {
		block := raw[r*quant.BlockQ8_0Size:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // fp16 1.0
		for i := range quant.QK8_0 {
			v := int8(i + 1)
			if r == 1 {
				v = int8(quant.QK8_0 - i)
			}
			block[2+i] = byte(v)
		}
	}

	want, err := ops.MatMulVecQ8_0(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	got, err := b.MatMulVecQ8_0Cached("test", raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("строка %d: получили %v, ожидали %v", i, got[i], want[i])
		}
	}

	// путь к графу воспроизведения
	got2, err := b.MatMulVecQ8_0Cached("test", raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got2[i]-want[i])) > 1e-3 {
			t.Fatalf("replay строка %d: получили %v, ожидали %v", i, got2[i], want[i])
		}
	}
}

func TestMatMulVecQ4_0GPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}
	defer b.Close()

	if !b.hasQ4 {
		t.Skip("CUDA q4_0 kernel недоступен")
	}

	rows, cols := 2, 32
	vec := make([]float32, cols)
	for i := range cols {
		vec[i] = 1
	}

	// Q4_0: scale fp16=1.0, qs pack nibbles for values 0..15 then 0..15 -> dequant (q-8)
	raw := make([]byte, rows*(cols/quant.QK4_0)*quant.BlockQ4_0Size)
	for r := range rows {
		block := raw[r*quant.BlockQ4_0Size:]
		binary.LittleEndian.PutUint16(block[0:2], 0x3c00) // fp16 1.0
		for i := 0; i < quant.QK4_0/2; i++ {
			lo := byte(i % 16)
			hi := byte((i + 8) % 16)
			if r == 1 {
				lo = byte((15 - i) % 16)
				hi = byte((7 - i + 16) % 16)
			}
			block[2+i] = lo | (hi << 4)
		}
	}

	want, err := ops.MatMulVecQ4_0(raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	got, err := b.MatMulVecQ4_0Cached("test-q4", raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("строка %d: получили %v, ожидали %v", i, got[i], want[i])
		}
	}

	got2, err := b.MatMulVecQ4_0Cached("test-q4", raw, rows, cols, vec)
	if err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(got2[i]-want[i])) > 1e-3 {
			t.Fatalf("replay строка %d: получили %v, ожидали %v", i, got2[i], want[i])
		}
	}
}

func TestRMSNormGPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA недоступна:", err)
	}

	if !b.hasRMS {
		t.Skip("CUDA rmsnorm kernel недоступен")
	}

	defer b.Close()

	x := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	dst := make([]float32, len(x))
	want := make([]float32, len(x))
	if err := ops.RMSNormInto(want, x, weight, 1e-6); err != nil {
		t.Fatal(err)
	}

	if err := b.RMSNormInto(dst, x, weight, 1e-6); err != nil {
		t.Fatal(err)
	}

	for i := range want {
		if math.Abs(float64(dst[i]-want[i])) > 1e-4 {
			t.Fatalf("элемент %d: получили %v, ожидали %v", i, dst[i], want[i])
		}
	}
}
