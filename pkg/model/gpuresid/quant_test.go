package gpuresid

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/quant"
)

func TestQuantBytes(t *testing.T) {
	cases := []struct {
		t    format.GGML
		rows int
		cols int
		want int
	}{
		{format.GgmlQ8_0, 4, 64, 4 * (64 / quant.QK8_0) * quant.BlockQ8_0Size},
		{format.GgmlQ4_0, 2, 64, 2 * (64 / quant.QK4_0) * quant.BlockQ4_0Size},
		{format.GgmlQ4_K, 3, 512, 3 * (512 / quant.QK_K) * quant.BlockQ4_KSize},
		{format.GgmlQ6_K, 1, 256, (256 / quant.QK_K) * quant.BlockQ6_KSize},
	}

	for _, c := range cases {
		got, err := QuantBytes(c.t, c.rows, c.cols)
		if err != nil {
			t.Fatalf("%s: %v", c.t, err)
		}

		if got != c.want {
			t.Fatalf("%s: %d байт, ожидали %d", c.t, got, c.want)
		}
	}
}

// expert slice must require cols divisible by block size and known type
func TestQuantBytesErrors(t *testing.T) {
	if _, err := QuantBytes(format.GgmlQ4_K, 2, 300); err == nil {
		t.Fatal("ожидали ошибку на cols не кратном QK_K")
	}

	if _, err := QuantBytes(format.GgmlQ2_K, 2, 256); err == nil {
		t.Fatal("ожидали ошибку на неподдерживаемом типе")
	}

	if _, err := QuantBytes(format.GgmlQ8_0, 0, 64); err == nil {
		t.Fatal("ожидали ошибку на rows=0")
	}
}
