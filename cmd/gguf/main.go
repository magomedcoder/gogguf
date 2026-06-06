package main

import (
	"fmt"
	"os"
)

const usage = `gguf - система запуска GGUF-моделей на Go (gguf.go)

Использование:
  gguf inspect файл.gguf просмотр метаданных и тензоров
  gguf info -m файл.gguf краткая информация о модели
`

// main - точка входа CLI gguf
func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(1)
	}

	switch os.Args[1] {
	case "inspect":
		if len(os.Args) != 3 {
			fmt.Fprintf(os.Stderr, "использование: gguf inspect файл.gguf\n")
			os.Exit(1)
		}
		if err := runInspect(os.Args[2]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "info":
		if err := runInfo(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "неизвестная команда: %q\n\n", os.Args[1])
		fmt.Print(usage)
		os.Exit(1)
	}
}
