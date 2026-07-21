package debug

import (
	"math"
	"testing"
)

func TestDiffLogits(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{1, 2.5, 3}
	st := DiffLogits(a, b, 0.1)
	if st.N != 3 || st.MaxAbsIdx != 1 {
		t.Fatalf("stats = %+v", st)
	}

	if math.Abs(st.MaxAbs-0.5) > 1e-9 {
		t.Fatalf("MaxAbs = %v", st.MaxAbs)
	}

	if st.OverTol != 1 {
		t.Fatalf("OverTol = %d", st.OverTol)
	}
}

func TestLogSoftmaxInPlace(t *testing.T) {
	x := []float32{0, 0, 0}
	dst := make([]float32, 3)
	LogSoftmaxInPlace(dst, x)
	want := float32(math.Log(1.0 / 3.0))
	for i, v := range dst {
		if math.Abs(float64(v-want)) > 1e-5 {
			t.Fatalf("dst[%d]=%v want %v", i, v, want)
		}
	}
}

func TestWriteReadLogitsBin(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/sample"
	meta := LogitsMeta{
		Model:  "test",
		Prompt: "Hi",
		Tokens: []int{1},
	}
	logits := []float32{1.5, -2, 3.25}
	if err := SaveLogitsDump(path, meta, logits); err != nil {
		t.Fatal(err)
	}

	gotMeta, got, err := LoadLogitsDump(path)
	if err != nil {
		t.Fatal(err)
	}

	if gotMeta.Model != "test" || gotMeta.Vocab != 3 {
		t.Fatalf("meta = %+v", gotMeta)
	}

	for i := range logits {
		if got[i] != logits[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], logits[i])
		}
	}
}
