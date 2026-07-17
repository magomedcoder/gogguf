package main

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/hf"
)

// resolveModelPath возвращает локальный путь к GGUF из -m или -hf.
// Ровно один из аргументов должен быть задан.
func resolveModelPath(m, hfRepo string) (string, error) {
	switch {
	case m != "" && hfRepo != "":
		return "", fmt.Errorf("укажите либо -m, либо -hf, не оба")
	case m != "":
		return m, nil
	case hfRepo != "":
		return hf.Resolve(hfRepo)
	default:
		return "", fmt.Errorf("укажите модель через -m файл.gguf или -hf owner/repo[:quant]")
	}
}
