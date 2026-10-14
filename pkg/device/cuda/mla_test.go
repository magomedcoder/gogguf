//go:build cuda

package cuda

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/device/mlaspec"
)

// mlaCPUReference mirrors deepseek2.mlaAttn absorbed path (absorb via k_b, attention, v_b projection).
func mlaCPUReference(t *testing.T, q, kb, vb, k, v []float32, seq int, p mlaspec.Params) []float32 {
	t.Helper()

	qk := p.KDim()
	headK := p.Nope + p.Rope
	qAbs := make([]float32, p.NHeads*qk)
	for h := 0; h < p.NHeads; h++ {
		src := q[h*headK : (h+1)*headK]
		wOff := h * p.Nope * p.KVLora
		if err := compute.MatMulColMajorInto(kb[wOff:wOff+p.Nope*p.KVLora], p.Nope, p.KVLora, src[:p.Nope], qAbs[h*qk:h*qk+p.KVLora]); err != nil {
			t.Fatal(err)
		}

		copy(qAbs[h*qk+p.KVLora:(h+1)*qk], src[p.Nope:])
	}

	out := make([]float32, p.OutDim())
	if err := compute.AttentionMLAAbsorbedInto(out, qAbs, k, v, make([]float32, seq), vb, seq, p.NHeads, p.KVLora, p.Rope, p.VHead, p.Scale); err != nil {
		t.Fatal(err)
	}

	return out
}

// TestMLAAttentionMatchesCPU: device MLA cache + mla_attn kernel vs CPU absorbed MLA (seq crosses the 256-position chunk; kvLora=300 uses two accumulators per thread).
func TestMLAAttentionMatchesCPU(t *testing.T) {
	b, err := Open()
	if err != nil {
		t.Skip("CUDA unavailable:", err)
	}
	defer b.Close()

	if !b.hasMLA {
		t.Skip("mla_attn kernel unavailable")
	}

	cases := []struct {
		name string
		seq  int
		p    mlaspec.Params
	}{
		{"small", 5, mlaspec.Params{
			NHeads: 2,
			Nope:   8,
			KVLora: 16,
			Rope:   4,
			VHead:  8,
			Scale:  0.25,
		}},
		{"multichunk", 300, mlaspec.Params{
			NHeads: 4,
			Nope:   32,
			KVLora: 300,
			Rope:   16,
			VHead:  24,
			Scale:  0.0625,
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.p
			rng := rand.New(rand.NewSource(7))
			rnd := func(n int, s float32) []float32 {
				o := make([]float32, n)
				for i := range o {
					o[i] = (rng.Float32()*2 - 1) * s
				}
				return o
			}

			kb := rnd(p.NHeads*p.KVLora*p.Nope, 0.3)
			vb := rnd(p.NHeads*p.VHead*p.KVLora, 0.3)
			k := rnd(tc.seq*p.KDim(), 1)
			v := make([]float32, tc.seq*p.KVLora)
			for i := 0; i < tc.seq; i++ {
				copy(v[i*p.KVLora:(i+1)*p.KVLora], k[i*p.KDim():i*p.KDim()+p.KVLora])
			}
			q := rnd(p.QDim(), 1)

			if err := b.MLACacheInit(2, tc.seq+4, p); err != nil {
				t.Fatal(err)
			}

			for i := 0; i < tc.seq; i++ {
				if err := b.MLACacheAppend(1, i, k[i*p.KDim():(i+1)*p.KDim()], v[i*p.KVLora:(i+1)*p.KVLora]); err != nil {
					t.Fatal(err)
				}
			}

			loads := 0
			load := func(name string) ([]float32, error) {
				loads++
				switch name {
				case "kb":
					return kb, nil
				case "vb":
					return vb, nil
				}
				return nil, errors.New("unknown weight " + name)
			}

			want := mlaCPUReference(t, q, kb, vb, k, v, tc.seq, p)
			got := make([]float32, p.OutDim())
			for rep := 0; rep < 2; rep++ {
				if err := b.MLAAttention(1, got, q, tc.seq, p, "kb", "vb", load); err != nil {
					t.Fatal(err)
				}

				for i := range want {
					if math.Abs(float64(got[i]-want[i])) > 5e-4*(1+math.Abs(float64(want[i]))) {
						t.Fatalf("rep=%d [%d] gpu=%v cpu=%v", rep, i, got[i], want[i])
					}
				}
			}

			if loads != 2 {
				t.Fatalf("k_b/v_b must upload once, loader called %d times", loads)
			}
		})
	}
}
