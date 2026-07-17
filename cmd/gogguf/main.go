package main

import (
	"fmt"
	"os"
)

const usage = `GoGGUF - система запуска GGUF-моделей на Go

Использование:
  gogguf inspect файл.gguf                    просмотр метаданных и тензоров
  gogguf info -m файл.gguf                    краткая информация о модели
  gogguf run -m файл.gguf -p "..."            генерация текста
  gogguf run -hf owner/repo[:quant] -p "..."  скачать с Hugging Face и запустить
  gogguf run -m файл.gguf -i                  интерактивный режим (REPL)
  gogguf serve -m файл.gguf                   HTTP API (SSE streaming)
  gogguf serve -hf owner/repo[:quant]         HTTP API с моделью с Hugging Face
`

// main - точка входа CLI gogguf
func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "inspect":
		if len(os.Args) != 3 {
			fmt.Fprintf(os.Stderr, "использование: gogguf inspect файл.gguf\n")
			os.Exit(1)
		}
		err = runInspect(os.Args[2])
	case "info":
		err = runInfo(os.Args[2:])
	case "run":
		err = runRun(os.Args[2:])
	case "serve":
		err = runServe(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "неизвестная команда: %q\n\n", os.Args[1])
		fmt.Print(usage)
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
