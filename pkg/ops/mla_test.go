package ops

import (
	"math"
	"testing"
)

func TestAttentionMLAAbsorbedIdentity(t *testing.T) {
	// 1 head, kvLora=2, rope=0, vHead=2; wv_b = identity
	nHeads, kvLora, rope, vHead := 1, 2, 0, 2
	q := []float32{1, 0}
	k := []float32{1, 0} // seq=1
	v := []float32{3, 4}
	// Identity per head in col-major: out[v]=Σ_k W[k+kv*v]*ctx; W[:,0]=[1,0], W[:,1]=[0,1]
	wv := []float32{1, 0, 0, 1}
	dst := make([]float32, vHead)
	scores := make([]float32, 1)
	if err := AttentionMLAAbsorbedInto(dst, q, k, v, scores, wv, 1, nHeads, kvLora, rope, vHead, 1); err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(dst[0]-3)) > 1e-5 || math.Abs(float64(dst[1]-4)) > 1e-5 {
		t.Fatalf("dst=%v, ожидали [3,4]", dst)
	}
}

func TestMatMulColMajorInto(t *testing.T) {
	// W = [[1,2],[3,4]] by columns: col0=[1,3], col1=[2,4] -> [1,3,2,4]
	// out[0]=1*5+3*6=23, out[1]=2*5+4*6=34
	w := []float32{1, 3, 2, 4}
	vec := []float32{5, 6}
	out := make([]float32, 2)
	if err := MatMulColMajorInto(w, 2, 2, vec, out); err != nil {
		t.Fatal(err)
	}

	if out[0] != 23 || out[1] != 34 {
		t.Fatalf("out=%v", out)
	}
}
