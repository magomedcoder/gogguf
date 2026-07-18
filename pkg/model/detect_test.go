package model

import (
	"testing"

	"github.com/magomedcoder/gogguf/pkg/format"
)

func TestIsMistralModel(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture": "llama",
			"general.name":         "mistralai_mistral-7b-instruct-v0.2",
		},
	}
	if !isMistralModel(r) {
		t.Fatal("ожидали определение mistral по имени")
	}

	r2 := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture":           "llama",
			"general.name":                   "Llama-3.2-1B",
			"llama.attention.sliding_window": int32(4096),
		},
	}
	if !isMistralModel(r2) {
		t.Fatal("ожидали определение mistral по sliding_window")
	}

	r3 := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture": "llama",
			"general.name":         "Llama-3.2-1B",
		},
	}
	if isMistralModel(r3) {
		t.Fatal("модель llama не должна определяться как mistral")
	}

	r4 := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture": "mistral",
		},
	}
	if !isMistralModel(r4) {
		t.Fatal("ожидали архитектуру mistral")
	}
}
