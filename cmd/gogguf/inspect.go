package main

import (
	"encoding/binary"
	"fmt"
	"sort"
	"strings"

	"github.com/magomedcoder/gogguf"
)

// runInspect prints metadata and the file's tensor list.
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

	fmt.Printf("Byte order: %s\n", byteOrderLabel(r.ByteOrder))
	fmt.Printf("GGUF version: %d\n", r.Version)

	for _, k := range keys {
		printMetadata(k, r.Metadata[k])
	}

	for _, t := range r.Tensors {
		dims := make([]string, len(t.Dimensions))
		for i, d := range t.Dimensions {
			dims[i] = fmt.Sprintf("%d", d)
		}
		fmt.Printf("Tensor: %s: %s [%s] (%d bytes)\n", t.Name, t.Type, strings.Join(dims, "x"), t.Size())
	}

	return nil
}

// byteOrderLabel returns a readable byte order name.
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

// printMetadata prints one metadata field.
func printMetadata(name string, v any) {
	switch vv := v.(type) {
	case []uint8, []int8, []uint16, []int16, []uint32, []int32, []float32, []bool, []string, []uint64, []int64, []float64:
		fmt.Printf("Metadata: %s: [%T len=%d]\n", name, vv, sliceLen(vv))
	default:
		fmt.Printf("Metadata: %s: %v\n", name, v)
	}
}

// sliceLen returns the length of a slice of any supported type.
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
