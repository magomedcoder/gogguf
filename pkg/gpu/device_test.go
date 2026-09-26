package gpu

import (
	"errors"
	"slices"
	"testing"
)

func TestParseDeviceList(t *testing.T) {
	cases := []struct {
		in   string
		want []int
	}{
		{"", nil},
		{"0", []int{0}},
		{"1", []int{1}},
		{"0,1", []int{0, 1}},
		{" 2 , 0 ", []int{2, 0}},
		{"1,1,0", []int{1, 0}}, // дубликаты убираются
	}

	for _, tc := range cases {
		got, err := ParseDeviceList(tc.in)
		if err != nil {
			t.Fatalf("ParseDeviceList(%q): %v", tc.in, err)
		}

		if !slices.Equal(got, tc.want) {
			t.Fatalf("ParseDeviceList(%q) = %v, ожидали %v", tc.in, got, tc.want)
		}
	}
}

func TestParseDeviceListErrors(t *testing.T) {
	if _, err := ParseDeviceList("-1"); !errors.Is(err, ErrInvalidDevice) {
		t.Fatalf("ParseDeviceList(-1) = %v, ожидали ErrInvalidDevice", err)
	}

	if _, err := ParseDeviceList("gpu0"); err == nil {
		t.Fatal("ParseDeviceList(gpu0): ожидали ошибку разбора")
	}
}

func TestParseTensorSplit(t *testing.T) {
	got, err := ParseTensorSplit("0.6, 0.4")
	if err != nil {
		t.Fatalf("ParseTensorSplit: %v", err)
	}

	if !slices.Equal(got, []float64{0.6, 0.4}) {
		t.Fatalf("ParseTensorSplit = %v", got)
	}

	if got, err := ParseTensorSplit(""); err != nil || got != nil {
		t.Fatalf("ParseTensorSplit(\"\") = %v, %v", got, err)
	}

	if _, err := ParseTensorSplit("0,0"); err == nil {
		t.Fatal("ParseTensorSplit(0,0): ожидали ошибку нулевой суммы")
	}

	if _, err := ParseTensorSplit("-1,2"); err == nil {
		t.Fatal("ParseTensorSplit(-1,2): ожидали ошибку отрицательной доли")
	}
}

func TestParseDevices(t *testing.T) {
	devs, split, err := ParseDevices("0,1", "24,4")
	if err != nil {
		t.Fatalf("ParseDevices: %v", err)
	}

	if !slices.Equal(devs, []int{0, 1}) || !slices.Equal(split, []float64{24, 4}) {
		t.Fatalf("ParseDevices = %v, %v", devs, split)
	}

	// -tensor-split без -dev задаёт устройства 0..N-1
	devs, split, err = ParseDevices("", "0.5,0.3,0.2")
	if err != nil {
		t.Fatalf("ParseDevices без -dev: %v", err)
	}

	if !slices.Equal(devs, []int{0, 1, 2}) || len(split) != 3 {
		t.Fatalf("ParseDevices без -dev = %v, %v", devs, split)
	}

	// Пустые флаги: устройство по умолчанию
	if devs, split, err = ParseDevices("", ""); err != nil || devs != nil || split != nil {
		t.Fatalf("ParseDevices(\"\", \"\") = %v, %v, %v", devs, split, err)
	}

	if _, _, err = ParseDevices("0,1", "0.5,0.3,0.2"); err == nil {
		t.Fatal("ParseDevices: ожидали ошибку несовпадения длин")
	}
}

func TestLayerSplitBounds(t *testing.T) {
	cases := []struct {
		layers, n int
		split     []float64
		want      []int
	}{
		{28, 1, nil, []int{0, 28}},
		{28, 2, nil, []int{0, 14, 28}},
		{28, 3, nil, []int{0, 10, 19, 28}},
		{28, 2, []float64{0.75, 0.25}, []int{0, 21, 28}},
		{28, 2, []float64{24, 4}, []int{0, 24, 28}},
		{1, 2, nil, []int{0, 1, 1}},
		{0, 2, nil, []int{0, 0, 0}},
		{28, 2, []float64{0, 0}, []int{0, 14, 28}}, // нулевая сумма = равномерно
	}

	for _, tc := range cases {
		got := LayerSplitBounds(tc.layers, tc.n, tc.split)
		if !slices.Equal(got, tc.want) {
			t.Fatalf("LayerSplitBounds(%d, %d, %v) = %v, ожидали %v", tc.layers, tc.n, tc.split, got, tc.want)
		}
	}

	if got := LayerSplitBounds(28, 0, nil); got != nil {
		t.Fatalf("LayerSplitBounds(28, 0) = %v, ожидали nil", got)
	}
}
