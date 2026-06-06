package main

import (
	"flag"
	"fmt"

	"github.com/magomedcoder/gguf.go"
)

// runInfo выводит краткую информацию о GGUF-файле
func runInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	modelPath := fs.String("m", "", "путь к файлу GGUF")
	if err := fs.Parse(args); err != nil {
		return err
	}
	
	if *modelPath == "" {
		return fmt.Errorf("использование: gguf info -m файл.gguf")
	}

	r, err := gguf.OpenFile(*modelPath)
	if err != nil {
		return err
	}

	arch, _ := r.Metadata.String("general.architecture")
	name, _ := r.Metadata.String("general.name")

	fmt.Printf("Файл:%s\n", *modelPath)
	fmt.Printf("Версия GGUF: %d\n", r.Version)
	fmt.Printf("Архитектура: %s\n", arch)
	if name != "" {
		fmt.Printf("Имя модели: %s\n", name)
	}
	fmt.Printf("Тензоров: %d\n", len(r.Tensors))
	fmt.Printf("Размер весов: %.1f MB\n", float64(r.TensorSize())/1e6)

	if arch != "" {
		ctxKey := arch + ".context_length"
		if ctx, err := r.Metadata.Int(ctxKey); err == nil {
			fmt.Printf("Контекст: %d токенов\n", ctx)
		}
	}

	return nil
}
