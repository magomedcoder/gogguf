package main

import (
	"encoding/binary"
	"fmt"
	"sort"
	"strings"

	"github.com/magomedcoder/gogguf"
)

// runInspect выводит метаданные и список тензоров файла
func runInspect(path string) error {
	r, err := gogguf.OpenFile(path)
	if err != nil {
		return err
	}

	keys := make([]string, 0, len(r.Metadata))
	for k := range r.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	fmt.Printf("Порядок байт: %s\n", byteOrderLabel(r.ByteOrder))
	fmt.Printf("Версия GGUF: %d\n", r.Version)

	for _, k := range keys {
		printMetadata(k, r.Metadata[k])
	}

	for _, t := range r.Tensors {
		dims := make([]string, len(t.Dimensions))
		for i, d := range t.Dimensions {
			dims[i] = fmt.Sprintf("%d", d)
		}
		fmt.Printf("Тензор: %s: %s [%s] (%d байт)\n", t.Name, t.Type, strings.Join(dims, "x"), t.Size())
	}

	return nil
}

// byteOrderLabel возвращает читаемое имя порядка байт
func byteOrderLabel(o binary.ByteOrder) string {
	switch o {
	case binary.LittleEndian:
		return "little-endian"
	case binary.BigEndian:
		return "big-endian"
	default:
		return o.String()
	}
}

// printMetadata выводит одно поле метаданных
func printMetadata(name string, v any) {
	switch vv := v.(type) {
	case []uint8, []int8, []uint16, []int16, []uint32, []int32, []float32, []bool, []string, []uint64, []int64, []float64:
		fmt.Printf("Метаданные: %s: [%T len=%d]\n", name, vv, sliceLen(vv))
	default:
		fmt.Printf("Метаданные: %s: %v\n", name, v)
	}
}

// sliceLen возвращает длину слайса любого поддерживаемого типа
func sliceLen(v any) int {
	switch s := v.(type) {
	case []uint8:
		return len(s)
	case []int8:
		return len(s)
	case []uint16:
		return len(s)
	case []int16:
		return len(s)
	case []uint32:
		return len(s)
	case []int32:
		return len(s)
	case []float32:
		return len(s)
	case []bool:
		return len(s)
	case []string:
		return len(s)
	case []uint64:
		return len(s)
	case []int64:
		return len(s)
	case []float64:
		return len(s)
	default:
		return 0
	}
}
