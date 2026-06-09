package runtime

import (
	"github.com/magomedcoder/gguf.go"
	"github.com/magomedcoder/gguf.go/model"
)

// Engine загружает GGUF-модель для inference
type Engine struct {
	Model model.Model
}

// Load открывает GGUF-файл и загружает модель
func Load(path string) (*Engine, error) {
	r, err := gguf.OpenFile(path)
	if err != nil {
		return nil, err
	}

	m, err := model.Load(r)
	if err != nil {
		return nil, err
	}

	return &Engine{
		Model: m,
	}, nil
}

// ForwardTokenIDs выполняет forward pass для token IDs
func (e *Engine) ForwardTokenIDs(tokens []int, startPos int) ([]float32, error) {
	return e.Model.Forward(tokens, startPos)
}
