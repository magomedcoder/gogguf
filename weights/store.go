package weights

import (
	"fmt"

	"github.com/magomedcoder/gguf.go"
	"github.com/magomedcoder/gguf.go/quant"
	"github.com/magomedcoder/gguf.go/tensor"
)

// Store хранит тензоры модели, загружаемые лениво по имени
type Store struct {
	reader *gguf.Reader
	raw    map[string][]byte
}

// New создаёт хранилище весов поверх GGUF-reader
func New(r *gguf.Reader) *Store {
	return &Store{
		reader: r,
		raw:    make(map[string][]byte),
	}
}

// Reader возвращает исходный GGUF-reader
func (s *Store) Reader() *gguf.Reader {
	return s.reader
}

// Raw возвращает сырые байты тензора (с кешированием)
func (s *Store) Raw(name string) ([]byte, error) {
	if b, ok := s.raw[name]; ok {
		return b, nil
	}

	info, err := s.reader.TensorInfo(name)
	if err != nil {
		return nil, err
	}

	b, err := tensor.LoadRaw(info)
	if err != nil {
		return nil, fmt.Errorf("weights %q: %w", name, err)
	}

	s.raw[name] = b

	return b, nil
}

// Info возвращает описание тензора
func (s *Store) Info(name string) (*gguf.TensorInfo, error) {
	return s.reader.TensorInfo(name)
}

// Floats загружает и деквантизирует тензор (norm weights и т.п.)
func (s *Store) Floats(name string) ([]float32, error) {
	info, err := s.reader.TensorInfo(name)
	if err != nil {
		return nil, err
	}

	raw, err := s.Raw(name)
	if err != nil {
		return nil, err
	}

	return quant.ToFloat32(info.Type, raw, int(info.ValuesCount()))
}
