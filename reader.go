package gguf

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

var errReaderAtRequired = errors.New("gguf: источник данных должен реализовывать io.ReaderAt")

// Reader - читатель файлов GGUF
type Reader struct {
	r io.ReadSeeker

	// ByteOrder - порядок байт файла GGUF
	// Пакет не выполняет перестановку байт для данных тензоров
	ByteOrder binary.ByteOrder

	Version  int
	Metadata Metadata
	Tensors  []TensorInfo

	tensorOffset int64
}

// readString читает строку GGUF: длина + байты
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

// readMetaDataValueScalar читает одно скалярное значение метаданных
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
			return nil, fmt.Errorf("недопустимое значение bool: %d", i)
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
		return nil, fmt.Errorf("недопустимый скалярный тип: %d", typ)
	}
}

// readMetaDataValueArray читает массив однотипных значений метаданных
func readMetaDataValueArray[T readables](r *Reader, length uint64) ([]T, error) {
	a := make([]T, length)
	for i := uint64(0); i < length; i++ {
		v, err := read[T](r.r, r.ByteOrder)
		if err != nil {
			return nil, err
		}
		a[i] = v
	}
	return a, nil
}

// readMetaValue читает значение метаданных (скаляр или массив)
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
				return nil, fmt.Errorf("недопустимое значение bool: %d", v)
			}
			b[i] = v == 1
		}
		return b, nil
	case String:
		a := make([]string, length)
		for i := uint64(0); i < length; i++ {
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
		return nil, fmt.Errorf("неподдерживаемый тип массива: %d", aType)
	}
}

// OpenFile открывает файл GGUF
func OpenFile(filename string) (*Reader, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	return Open(f)
}

// Open открывает файл GGUF из r. r должен быть позиционирован в начале файла и реализовывать io.ReaderAt для чтения данных тензоров
func Open(readSeeker io.ReadSeeker) (*Reader, error) {
	var buf [4]byte
	if _, err := readSeeker.Read(buf[:]); err != nil {
		return nil, err
	}
	if !bytes.Equal(buf[:], []byte(magic)) {
		return nil, fmt.Errorf("не файл GGUF, неизвестная магическая последовательность: %q", buf)
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
		return nil, fmt.Errorf("недопустимая версия: %d (поддерживаются 2 и 3)", version)
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
	for i := uint64(0); i < metadataCount; i++ {
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
			return nil, fmt.Errorf("недопустимый тип выравнивания: %T", a)
		}
		alignment = int64(v)
	}

	r.Tensors = make([]TensorInfo, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
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
		for j := uint32(0); j < nDimensions; j++ {
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

// TensorInfo возвращает информацию о тензоре с указанным именем
func (r *Reader) TensorInfo(name string) (*TensorInfo, error) {
	for i := range r.Tensors {
		if r.Tensors[i].Name == name {
			return &r.Tensors[i], nil
		}
	}
	return nil, fmt.Errorf("тензор %q не найден", name)
}

// TensorSize возвращает суммарный размер всех тензоров в файле
func (r *Reader) TensorSize() int64 {
	var size int64
	for _, t := range r.Tensors {
		size += t.Size()
	}
	return size
}
