package format

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	magic            = "GGUF"
	defaultAlignment = int64(32)
)

var errReaderAtRequired = errors.New("gguf: data source must implement io.ReaderAt")

// Reader - GGUF file reader
type Reader struct {
	r io.ReadSeeker

	// ByteOrder - GGUF file byte order
	// Package does not byte-swap tensor data
	ByteOrder binary.ByteOrder

	Version  int
	Metadata Metadata
	Tensors  []TensorInfo

	tensorOffset int64
}

// readString reads GGUF string: length + bytes
func (r *Reader) readString() (string, error) {
	length, err := read[uint64](r.r, r.ByteOrder)
	if err != nil {
		return "", err
	}

	data := make([]byte, length)
	if _, err = io.ReadFull(r.r, data); err != nil {
		return "", err
	}

	return strings.TrimSpace(string(data)), nil
}

// readMetaDataValueScalar reads one scalar metadata value
func (r *Reader) readMetaDataValueScalar(typ Type) (any, error) {
	switch typ {
	case Uint8:
		return read[uint8](r.r, r.ByteOrder)
	case Int8:
		return read[int8](r.r, r.ByteOrder)
	case Uint16:
		return read[uint16](r.r, r.ByteOrder)
	case Int16:
		return read[int16](r.r, r.ByteOrder)
	case Uint32:
		return read[uint32](r.r, r.ByteOrder)
	case Int32:
		return read[int32](r.r, r.ByteOrder)
	case Float32:
		return read[float32](r.r, r.ByteOrder)
	case Bool:
		i, err := read[uint8](r.r, r.ByteOrder)
		if err != nil {
			return nil, err
		}
		if i != 0 && i != 1 {
			return nil, fmt.Errorf("invalid bool value: %d", i)
		}
		return i == 1, nil
	case String:
		return r.readString()
	case Uint64:
		return read[uint64](r.r, r.ByteOrder)
	case Int64:
		return read[int64](r.r, r.ByteOrder)
	case Float64:
		return read[float64](r.r, r.ByteOrder)
	default:
		return nil, fmt.Errorf("invalid scalar type: %d", typ)
	}
}

// readMetaDataValueArray reads array of homogeneous metadata values
func readMetaDataValueArray[T readables](r *Reader, length uint64) ([]T, error) {
	a := make([]T, length)
	for i := range length {
		v, err := read[T](r.r, r.ByteOrder)
		if err != nil {
			return nil, err
		}
		a[i] = v
	}
	return a, nil
}

// readMetaValue reads metadata value (scalar or array)
func (r *Reader) readMetaValue() (any, error) {
	typ, err := read[Type](r.r, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	if typ != Array {
		return r.readMetaDataValueScalar(typ)
	}

	aType, err := read[Type](r.r, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	length, err := read[uint64](r.r, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	switch aType {
	case Uint8:
		return readMetaDataValueArray[uint8](r, length)
	case Int8:
		return readMetaDataValueArray[int8](r, length)
	case Uint16:
		return readMetaDataValueArray[uint16](r, length)
	case Int16:
		return readMetaDataValueArray[int16](r, length)
	case Uint32:
		return readMetaDataValueArray[uint32](r, length)
	case Int32:
		return readMetaDataValueArray[int32](r, length)
	case Float32:
		return readMetaDataValueArray[float32](r, length)
	case Bool:
		a, err := readMetaDataValueArray[uint8](r, length)
		if err != nil {
			return nil, err
		}
		b := make([]bool, length)
		for i, v := range a {
			if v != 0 && v != 1 {
				return nil, fmt.Errorf("invalid bool value: %d", v)
			}
			b[i] = v == 1
		}
		return b, nil
	case String:
		a := make([]string, length)
		for i := range length {
			v, err := r.readString()
			if err != nil {
				return nil, err
			}
			a[i] = v
		}
		return a, nil
	case Uint64:
		return readMetaDataValueArray[uint64](r, length)
	case Int64:
		return readMetaDataValueArray[int64](r, length)
	case Float64:
		return readMetaDataValueArray[float64](r, length)
	default:
		return nil, fmt.Errorf("unsupported array type: %d", aType)
	}
}

// OpenFile opens GGUF file
func OpenFile(filename string) (*Reader, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	return Open(f)
}

// Open opens GGUF file from r. r must be at file start and implement io.ReaderAt for tensor data reads
func Open(readSeeker io.ReadSeeker) (*Reader, error) {
	var buf [4]byte
	if _, err := readSeeker.Read(buf[:]); err != nil {
		return nil, err
	}
	if !bytes.Equal(buf[:], []byte(magic)) {
		return nil, fmt.Errorf("not a GGUF file, unknown magic sequence: %q", buf)
	}

	if _, err := readSeeker.Seek(3, io.SeekCurrent); err != nil {
		return nil, err
	}

	bigEndianMarker := int8(0)
	if err := binary.Read(readSeeker, binary.LittleEndian, &bigEndianMarker); err != nil {
		return nil, err
	}

	var byteOrder binary.ByteOrder = binary.LittleEndian
	if bigEndianMarker != 0 {
		byteOrder = binary.BigEndian
	}

	if _, err := readSeeker.Seek(-4, io.SeekCurrent); err != nil {
		return nil, err
	}

	version, err := read[uint32](readSeeker, byteOrder)
	if err != nil {
		return nil, err
	}
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("invalid version: %d (supported: 2 and 3)", version)
	}

	r := &Reader{
		r:         readSeeker,
		ByteOrder: byteOrder,
		Version:   int(version),
	}

	tensorCount, err := read[uint64](readSeeker, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	metadataCount, err := read[uint64](readSeeker, r.ByteOrder)
	if err != nil {
		return nil, err
	}

	r.Metadata = make(Metadata, metadataCount)
	for range metadataCount {
		name, err := r.readString()
		if err != nil {
			return nil, err
		}

		value, err := r.readMetaValue()
		if err != nil {
			return nil, err
		}

		if u, ok := value.(uint32); ok && name == "general.file_type" {
			value = Filetype(u)
		}

		r.Metadata[name] = value
	}

	alignment := defaultAlignment
	if a, found := r.Metadata["general.alignment"]; found {
		v, ok := a.(uint32)
		if !ok {
			return nil, fmt.Errorf("invalid alignment type: %T", a)
		}
		alignment = int64(v)
	}

	r.Tensors = make([]TensorInfo, tensorCount)
	for i := range tensorCount {
		r.Tensors[i].reader = r

		r.Tensors[i].Name, err = r.readString()
		if err != nil {
			return nil, err
		}

		nDimensions, err := read[uint32](readSeeker, r.ByteOrder)
		if err != nil {
			return nil, err
		}

		r.Tensors[i].Dimensions = make([]uint64, nDimensions)
		for j := range nDimensions {
			r.Tensors[i].Dimensions[j], err = read[uint64](readSeeker, r.ByteOrder)
			if err != nil {
				return nil, err
			}
		}

		typ, err := read[uint32](readSeeker, r.ByteOrder)
		if err != nil {
			return nil, err
		}
		r.Tensors[i].Type = GGML(typ)

		r.Tensors[i].Offset, err = read[uint64](readSeeker, r.ByteOrder)
		if err != nil {
			return nil, err
		}
	}

	current, err := readSeeker.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	r.tensorOffset = (current + alignment - 1) / alignment * alignment
	return r, nil
}

// TensorInfo returns info for tensor with given name
func (r *Reader) TensorInfo(name string) (*TensorInfo, error) {
	for i := range r.Tensors {
		if r.Tensors[i].Name == name {
			return &r.Tensors[i], nil
		}
	}
	return nil, fmt.Errorf("tensor %q not found", name)
}

// TensorSize returns total size of all tensors in file
func (r *Reader) TensorSize() int64 {
	var size int64
	for _, t := range r.Tensors {
		size += t.Size()
	}
	return size
}
