package weights

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/quant"
	"github.com/magomedcoder/gogguf/pkg/tensor"
)

// Store хранит тензоры модели, загружаемые лениво по имени
type Store struct {
	reader *format.Reader
	raw    map[string][]byte
	floats map[string][]float32
}

// New создаёт хранилище весов поверх GGUF-reader
func New(r *format.Reader) *Store {
	return &Store{
		reader: r,
		raw:    make(map[string][]byte),
		floats: make(map[string][]float32),
	}
}

// Reader возвращает исходный GGUF-reader
func (s *Store) Reader() *format.Reader {
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

	b, err := tensor.LoadRawView(info)
	if err != nil {
		return nil, fmt.Errorf("weights %q: %w", name, err)
	}

	s.raw[name] = b

	return b, nil
}

// Info возвращает описание тензора
func (s *Store) Info(name string) (*format.TensorInfo, error) {
	return s.reader.TensorInfo(name)
}

// Floats загружает и деквантизирует тензор (norm weights и т.п.)
func (s *Store) Floats(name string) ([]float32, error) {
	if f, ok := s.floats[name]; ok {
		return f, nil
	}

	info, err := s.reader.TensorInfo(name)
	if err != nil {
		return nil, err
	}

	raw, err := s.Raw(name)
	if err != nil {
		return nil, err
	}

	f, err := quant.ToFloat32(info.Type, raw, int(info.ValuesCount()))
	if err != nil {
		return nil, err
	}

	s.floats[name] = f
	return f, nil
}
