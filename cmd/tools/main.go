package main

import (
	"fmt"
	"os"
)

const usage = `GoGGUF tools - утилиты отладки и бенчмарка

Использование:
  tools bench -m файл.gguf -p "..."          бенчмарк prefill/decode/TTFT
  tools debugtok файл.gguf "промпт"          encode + top logits после prefill
  tools vocab файл.gguf                      конфиг и special tokens
  tools greedy -m файл.gguf --chat "..."     greedy decode (JSON token IDs)
  tools debuglayers -m файл.gguf -p "..."    послойный RMS + logits
  tools layerlogits -m файл.gguf -p "..."    greedy/top logits по слоям (fixture)
  tools dumplogits -m файл.gguf -p "..."     полный vocab logits -> .bin/.json
  tools comparelogits -a dump -b dump        сверка двух dump (или -m CPU vs GPU)
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "bench":
		err = runBench(os.Args[2:])
	case "debugtok":
		err = runDebugTok(os.Args[2:])
	case "vocab":
		err = runVocab(os.Args[2:])
	case "greedy":
		err = runGreedy(os.Args[2:])
	case "debuglayers":
		err = runDebugLayers(os.Args[2:])
	case "layerlogits":
		err = runLayerLogits(os.Args[2:])
	case "dumplogits":
		err = runDumpLogits(os.Args[2:])
	case "comparelogits":
		err = runCompareLogits(os.Args[2:])
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
