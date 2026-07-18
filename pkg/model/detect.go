package model

import (
	"strings"

	"github.com/magomedcoder/gogguf/pkg/format"
)

// isMistralModel определяет Mistral по architecture или метаданным (TheBloke: arch=llama)
func isMistralModel(r *format.Reader) bool {
	arch, err := r.Metadata.String("general.architecture")
	if err != nil {
		return false
	}

	if arch == "mistral" {
		return true
	}

	if arch != "llama" {
		return false
	}

	if name, err := r.Metadata.String("general.name"); err == nil {
		if strings.Contains(strings.ToLower(name), "mistral") {
			return true
		}
	}

	if _, err := r.Metadata.Int("llama.attention.sliding_window"); err == nil {
		return true
	}

	if _, err := r.Metadata.Int("mistral.attention.sliding_window"); err == nil {
		return true
	}

	return false
}
