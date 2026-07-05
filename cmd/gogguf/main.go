package main

import (
	"fmt"
	"os"
)

const usage = `GoGGUF - система запуска GGUF-моделей на Go

Использование:
  gogguf inspect файл.gguf          просмотр метаданных и тензоров
  gogguf info -m файл.gguf          краткая информация о модели
  gogguf run -m файл.gguf -p "..."  генерация текста
  gogguf run -m файл.gguf -i        интерактивный режим (REPL)
  gogguf serve -m файл.gguf         HTTP API (SSE streaming)
`

// main - точка входа CLI gogguf
func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(1)
	}

	switch os.Args[1] {
	case "inspect":
		if len(os.Args) != 3 {
			fmt.Fprintf(os.Stderr, "использование: gogguf inspect файл.gguf\n")
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
	case "run":
		if err := runRun(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "serve":
		if err := runServe(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "неизвестная команда: %q\n\n", os.Args[1])
		fmt.Print(usage)
		os.Exit(1)
	}
}
