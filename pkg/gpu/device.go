package gpu

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseDeviceList разбирает значение флага -dev: "1" или "0,1" (ordinal устройств после CUDA_VISIBLE_DEVICES). Пустая строка = устройство по умолчанию (nil)
func ParseDeviceList(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}

	parts := strings.Split(s, ",")
	devices := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}

		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("gpu: -dev %q: ожидался номер устройства: %w", p, err)
		}

		devices = append(devices, n)
	}

	return normalizeDeviceList(devices)
}

// ParseTensorSplit разбирает -tensor-split: "0.6,0.4" или "24,4" (веса пропорций слоёв)
func ParseTensorSplit(s string) ([]float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}

	parts := strings.Split(s, ",")
	split := make([]float64, 0, len(parts))
	var total float64
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}

		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, fmt.Errorf("gpu: -tensor-split %q: ожидалось число: %w", p, err)
		}

		if v < 0 {
			return nil, fmt.Errorf("gpu: -tensor-split %q: доля должна быть >= 0", p)
		}

		total += v
		split = append(split, v)
	}

	if len(split) == 0 {
		return nil, nil
	}

	if total <= 0 {
		return nil, fmt.Errorf("gpu: -tensor-split: сумма долей должна быть > 0")
	}

	return split, nil
}

// ParseDevices разбирает пару флагов -dev / -tensor-split в план offload.
// Пустой -dev с несколькими долями в -tensor-split означает устройства 0..N-1
func ParseDevices(devList, splitList string) (devices []int, split []float64, err error) {
	split, err = ParseTensorSplit(splitList)
	if err != nil {
		return nil, nil, err
	}

	devices, err = ParseDeviceList(devList)
	if err != nil {
		return nil, nil, err
	}

	if devices == nil && len(split) > 1 {
		devices = make([]int, len(split))
		for i := range split {
			devices[i] = i
		}
	}

	if len(split) > 0 && len(devices) != len(split) {
		return nil, nil, fmt.Errorf("gpu: -tensor-split из %d долей при %d устройствах в -dev", len(split), len(devices))
	}

	return devices, split, nil
}

// normalizeDeviceList проверяет ordinal'ы и убирает дубликаты, сохраняя порядок
func normalizeDeviceList(devices []int) ([]int, error) {
	if len(devices) == 0 {
		return []int{0}, nil
	}

	seen := make(map[int]struct{}, len(devices))
	out := make([]int, 0, len(devices))
	for _, d := range devices {
		if d < 0 {
			return nil, fmt.Errorf("%w: %d", ErrInvalidDevice, d)
		}

		if _, dup := seen[d]; dup {
			continue
		}

		seen[d] = struct{}{}
		out = append(out, d)
	}

	return out, nil
}
