package gpu

import "testing"

func TestLayerOnGPU(t *testing.T) {
	total := 28
	cases := []struct {
		layer, ngl int
		want       bool
	}{
		{0, 0, false},
		{0, 10, true},
		{9, 10, true},
		{10, 10, false},
		{-1, 10, false},
		{0, total + 1, true},
	}

	for _, tc := range cases {
		got := LayerOnGPU(tc.layer, tc.ngl, total)
		if got != tc.want {
			t.Fatalf("LayerOnGPU(%d, %d, %d) = %v, ожидали %v",
				tc.layer, tc.ngl, total, got, tc.want)
		}
	}
}

func TestOpenCUDAStub(t *testing.T) {
	_, err := OpenCUDA()
	if err != ErrUnavailable {
		t.Fatalf("OpenCUDA() = %v, ожидали ErrUnavailable", err)
	}
}
