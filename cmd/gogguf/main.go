package main

import (
	"fmt"
	"os"
)

const usage = `GoGGUF - GGUF model runtime in Go

Usage:
  gogguf inspect file.gguf                    view metadata and tensors
  gogguf info -m file.gguf                    brief model info
  gogguf run -m file.gguf -p "..."            text generation
  gogguf run -hf owner/repo[:quant] -p "..."  download from Hugging Face and run
  gogguf run -m file.gguf -i                  interactive mode (REPL)
  gogguf serve -m file.gguf                   HTTP API (SSE streaming)
  gogguf serve -hf owner/repo[:quant]         HTTP API with Hugging Face model
`

// main is the gogguf CLI entry point.
func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "inspect":
		if len(os.Args) != 3 {
			fmt.Fprintf(os.Stderr, "usage: gogguf inspect file.gguf\n")
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
		fmt.Fprintf(os.Stderr, "unknown command: %q\n\n", os.Args[1])
		fmt.Print(usage)
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
