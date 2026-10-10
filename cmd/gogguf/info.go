package main

import (
	"flag"
	"fmt"

	"github.com/magomedcoder/gogguf"
)

// runInfo prints brief information about a GGUF file.
func runInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	modelPath := fs.String("m", "", "path to GGUF file")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" {
		return fmt.Errorf("usage: gogguf info -m file.gguf")
	}

	r, err := gogguf.OpenFile(*modelPath)
	if err != nil {
		return err
	}

	arch, _ := r.Metadata.String("general.architecture")
	name, _ := r.Metadata.String("general.name")

	fmt.Printf("File:%s\n", *modelPath)
	fmt.Printf("GGUF version: %d\n", r.Version)
	fmt.Printf("Architecture: %s\n", arch)
	if name != "" {
		fmt.Printf("Model name: %s\n", name)
	}
	fmt.Printf("Tensors: %d\n", len(r.Tensors))
	fmt.Printf("Weight size: %.1f MB\n", float64(r.TensorSize())/1e6)

	if arch != "" {
		ctxKey := arch + ".context_length"
		if ctx, err := r.Metadata.Int(ctxKey); err == nil {
			fmt.Printf("Context: %d tokens\n", ctx)
		}
	}

	return nil
}
