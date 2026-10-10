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
		t.Fatal("expected mistral detection by name")
	}

	r2 := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture":           "llama",
			"general.name":                   "Llama-3.2-1B",
			"llama.attention.sliding_window": int32(4096),
		},
	}
	if !isMistralModel(r2) {
		t.Fatal("expected mistral detection by sliding_window")
	}

	r3 := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture": "llama",
			"general.name":         "Llama-3.2-1B",
		},
	}
	if isMistralModel(r3) {
		t.Fatal("llama model must not be detected as mistral")
	}

	r4 := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture": "mistral",
		},
	}
	if !isMistralModel(r4) {
		t.Fatal("expected mistral architecture")
	}
}

func TestIsMixtralModel(t *testing.T) {
	r := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture": "llama",
			"llama.expert_count":   int32(8),
			"general.name":         "mixtral-8x7b-instruct-v0.1",
		},
	}

	if !isMixtralModel(r) {
		t.Fatal("expected Mixtral by expert_count")
	}

	r2 := &format.Reader{
		Metadata: format.Metadata{
			"general.architecture": "llama",
			"general.name":         "Llama-3.2-1B",
		},
	}

	if isMixtralModel(r2) {
		t.Fatal("Llama 3 must not be detected as Mixtral")
	}
}
