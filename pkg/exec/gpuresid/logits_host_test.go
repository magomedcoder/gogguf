package gpuresid

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/device"
)

func TestCanLogitsFromDeviceRequiresResidentBackend(t *testing.T) {
	if CanLogitsFromDevice(nil, 8, 8) {
		t.Fatal("nil backend must not enable GPU logits")
	}

	// CPU backend keeps hidden on host: no GPU logits pass
	if CanLogitsFromDevice(device.CPUBackend{}, 8, 8) {
		t.Fatal("CPU backend must not enable GPU logits")
	}
}

func TestAllLayersOffloaded(t *testing.T) {
	cases := []struct {
		ngl, layers int
		want        bool
	}{
		{0, 8, false},
		{4, 8, false},
		{8, 8, true},
		{99, 8, true},
		{8, 0, false},
	}

	for _, c := range cases {
		if got := AllLayersOffloaded(c.ngl, c.layers); got != c.want {
			t.Errorf("AllLayersOffloaded(%d,%d)=%v, want %v", c.ngl, c.layers, got, c.want)
		}
	}
}
