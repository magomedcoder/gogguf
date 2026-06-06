package gguf

import "io"

// TensorInfo представляет тензор в файле GGUF
type TensorInfo struct {
	reader *Reader

	Name       string
	Dimensions []uint64
	Type       GGML
	Offset     uint64
}

// Reader возвращает io.Reader для чтения данных тензора
// Читатель ограничен размером данных тензора и не меняет позицию исходного файла
func (t *TensorInfo) Reader() (io.Reader, error) {
	start := t.reader.tensorOffset + int64(t.Offset)
	size := t.Size()

	ra, ok := t.reader.r.(io.ReaderAt)
	if !ok {
		return nil, errReaderAtRequired
	}

	return io.NewSectionReader(ra, start, size), nil
}

// Size возвращает размер данных тензора в байтах
func (t *TensorInfo) Size() int64 {
	return t.Type.dataSize(t.Dimensions)
}
