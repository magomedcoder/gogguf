package model

import (
	"fmt"

	"github.com/magomedcoder/gguf.go"
	"github.com/magomedcoder/gguf.go/model/qwen3"
	"github.com/magomedcoder/gguf.go/weights"
)

// Model - интерфейс архитектуры для forward pass
type Model interface {
	Forward(tokenIDs []int, startPos int) ([]float32, error)
}

// Load загружает модель по полю general.architecture
func Load(r *gguf.Reader) (Model, error) {
	arch, err := r.Metadata.String("general.architecture")
	if err != nil {
		return nil, err
	}

	store := weights.New(r)

	switch arch {
	case "qwen3":
		return qwen3.Load(store)
	default:
		return nil, fmt.Errorf("model: архитектура %q не поддерживается", arch)
	}
}
