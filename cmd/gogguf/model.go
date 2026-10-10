package main

import (
	"fmt"

	"github.com/magomedcoder/gogguf/pkg/hf"
)

// resolveModelPath returns the local GGUF path from -m or -hf.
// Exactly one of the arguments must be set.
func resolveModelPath(m, hfRepo string) (string, error) {
	switch {
	case m != "" && hfRepo != "":
		return "", fmt.Errorf("specify either -m or -hf, not both")
	case m != "":
		return m, nil
	case hfRepo != "":
		return hf.Resolve(hfRepo)
	default:
		return "", fmt.Errorf("specify model via -m file.gguf or -hf owner/repo[:quant]")
	}
}
