package gpuresid

import (
	"github.com/magomedcoder/gogguf/pkg/dtype"
	"testing"

	"github.com/magomedcoder/gogguf/pkg/quant"
)

func TestQuantBytes(t *testing.T) {
	cases := []struct {
		t    dtype.DType
		rows int
		cols int
		want int
	}{
		{dtype.Q8_0, 4, 64, 4 * (64 / quant.QK8_0) * quant.BlockQ8_0Size},
		{dtype.Q4_0, 2, 64, 2 * (64 / quant.QK4_0) * quant.BlockQ4_0Size},
		{dtype.Q4_1, 2, 64, 2 * (64 / quant.QK4_1) * quant.BlockQ4_1Size},
		{dtype.Q5_0, 1, 32, quant.BlockQ5_0Size},
		{dtype.Q2_K, 2, 256, 2 * quant.BlockQ2_KSize},
		{dtype.Q4_K, 3, 512, 3 * (512 / quant.QK_K) * quant.BlockQ4_KSize},
		{dtype.Q6_K, 1, 256, (256 / quant.QK_K) * quant.BlockQ6_KSize},
		{dtype.Q8_K, 1, 256, quant.BlockQ8_KSize},
	}

	for _, c := range cases {
		got, err := QuantBytes(c.t, c.rows, c.cols)
		if err != nil {
			t.Fatalf("%s: %v", c.t, err)
		}

		if got != c.want {
			t.Fatalf("%s: %d bytes, expected %d", c.t, got, c.want)
		}
	}
}

// expert slice must require cols divisible by block size and known type
func TestQuantBytesErrors(t *testing.T) {
	if _, err := QuantBytes(dtype.Q4_K, 2, 300); err == nil {
		t.Fatal("expected error for cols not divisible by QK_K")
	}

	if _, err := QuantBytes(dtype.Q8_1, 2, 256); err == nil {
		t.Fatal("expected error for unsupported type")
	}

	if _, err := QuantBytes(dtype.Q8_0, 0, 64); err == nil {
		t.Fatal("expected error for rows=0")
	}
}
