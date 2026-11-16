package metal

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

// Go ports of the matmul_vec_q5_k / matmul_vec_q6_k inner loops in msl.go (same indexing, lane striding and field offsets over the packed pkg/quant layouts).
// They run everywhere, so the block arithmetic the MSL relies on (qh bit planes, 6-bit scale unpacking, Q6_K int8 scales) is checked against compute.MatMulVecQ5_K / Q6_K without a Mac; the MSL text itself is compiled only by the darwin runtime.

func mslScaleMinK4(j int, q []byte) (sc, mn uint32) {
	if j < 4 {
		return uint32(q[j]) & 63, uint32(q[j+4]) & 63
	}

	return uint32(q[j+4])&0xF | (uint32(q[j-4])>>6)<<4, uint32(q[j+4])>>4 | (uint32(q[j])>>6)<<4
}

func f16At(b []byte) float32 {
	return quant.FP16ToFP32(binary.LittleEndian.Uint16(b))
}

// mslModelQ5K emulates one output row of matmul_vec_q5_k with a simdgroup of sgw lanes
func mslModelQ5K(row []byte, x []float32, nb, sgw int) float32 {
	var sum float32
	for lane := range sgw {
		var lsum float32
		for sb := range nb {
			b := row[sb*quant.BlockQ5_KSize : (sb+1)*quant.BlockQ5_KSize]
			d, dmin := f16At(b[0:]), f16At(b[2:])
			scales, qh, qs := b[4:16], b[16:48], b[48:176]
			xs := x[sb*256:]
			for g := range 4 {
				s1, m1 := mslScaleMinK4(2*g, scales)
				s2, m2 := mslScaleMinK4(2*g+1, scales)
				d1, d2 := d*float32(s1), d*float32(s2)
				mm1, mm2 := dmin*float32(m1), dmin*float32(m2)
				u1, u2 := uint32(1)<<(2*g), uint32(2)<<(2*g)
				for l := lane; l < 32; l += sgw {
					q, h := uint32(qs[g*32+l]), uint32(qh[l])
					q1, q2 := q&0xF, q>>4
					if h&u1 != 0 {
						q1 += 16
					}

					if h&u2 != 0 {
						q2 += 16
					}

					lsum += (d1*float32(q1) - mm1) * xs[g*64+l]
					lsum += (d2*float32(q2) - mm2) * xs[g*64+32+l]
				}
			}
		}

		sum += lsum
	}

	return sum
}

// mslModelQ6K emulates one output row of matmul_vec_q6_k
func mslModelQ6K(row []byte, x []float32, nb, sgw int) float32 {
	var sum float32
	for lane := range sgw {
		var lsum float32
		for sb := range nb {
			b := row[sb*quant.BlockQ6_KSize : (sb+1)*quant.BlockQ6_KSize]
			d := f16At(b[208:])
			xs := x[sb*256:]
			for n := range 2 {
				ql, qh, sc := b[64*n:], b[128+32*n:], b[192+8*n:]
				xn := xs[128*n:]
				for l := lane; l < 32; l += sgw {
					is := l >> 4
					lo0, lo1, hi := int(ql[l]), int(ql[l+32]), int(qh[l])
					q1 := (lo0&0xF | ((hi>>0)&3)<<4) - 32
					q2 := (lo1&0xF | ((hi>>2)&3)<<4) - 32
					q3 := (lo0>>4 | ((hi>>4)&3)<<4) - 32
					q4 := (lo1>>4 | ((hi>>6)&3)<<4) - 32
					lsum += d * float32(int8(sc[is])) * float32(q1) * xn[l]
					lsum += d * float32(int8(sc[is+2])) * float32(q2) * xn[l+32]
					lsum += d * float32(int8(sc[is+4])) * float32(q3) * xn[l+64]
					lsum += d * float32(int8(sc[is+6])) * float32(q4) * xn[l+96]
				}
			}
		}

		sum += lsum
	}

	return sum
}

func TestMSLQ5KQ6KKernelArithmetic(t *testing.T) {
	for _, tc := range []struct {
		t     dtype.DType
		model func([]byte, []float32, int, int) float32
		ref   func([]byte, int, int, []float32) ([]float32, error)
	}{
		{dtype.Q5_K, mslModelQ5K, compute.MatMulVecQ5_K},
		{dtype.Q6_K, mslModelQ6K, compute.MatMulVecQ6_K},
	} {
		_, bs, _ := quantGeometry(tc.t)
		for _, sz := range []struct{ rows, blocks int }{{1, 1}, {3, 2}, {5, 4}} {
			cols := sz.blocks * quant.QK_K
			raw := fakeQuant(tc.t, sz.rows, cols, uint32(800+sz.rows))
			x := detFloats(cols, 801)

			want, err := tc.ref(raw, sz.rows, cols, x)
			if err != nil {
				t.Fatal(err)
			}

			for _, sgw := range []int{32, 16, 8} { // simdgroup width: lanes stride l by sgw
				got := make([]float32, sz.rows)
				for r := range sz.rows {
					got[r] = tc.model(raw[r*sz.blocks*bs:(r+1)*sz.blocks*bs], x, sz.blocks, sgw)
				}

				relEqual(t, fmt.Sprintf("%s model rows=%d blocks=%d sgw=%d", tc.t, sz.rows, sz.blocks, sgw), got, want, 1e-5)
			}
		}
	}
}
