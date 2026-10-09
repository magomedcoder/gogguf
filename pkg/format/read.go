package format

import (
	"encoding/binary"
	"io"
)

type readables interface {
	~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64
}

// read reads one value of type T from binary stream
func read[T readables](r io.Reader, byteOrder binary.ByteOrder) (T, error) {
	var v T
	err := binary.Read(r, byteOrder, &v)
	return v, err
}
