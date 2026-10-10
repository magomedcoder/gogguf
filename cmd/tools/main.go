package main

import (
	"fmt"
	"os"
)

const usage = `GoGGUF tools - debugging and benchmark utilities

Usage:
  tools bench -m file.gguf -p "..."          benchmark prefill/decode/TTFT
  tools debugtok file.gguf "prompt"          encode + top logits after prefill
  tools vocab file.gguf                      config and special tokens
  tools greedy -m file.gguf --chat "..."     greedy decode (JSON token IDs)
  tools debuglayers -m file.gguf -p "..."    per-layer RMS + logits
  tools layerlogits -m file.gguf -p "..."    greedy/top logits per layer (fixture)
  tools dumplogits -m file.gguf -p "..."     full vocab logits -> .bin/.json
  tools comparelogits -a dump -b dump        compare two dumps (or -m CPU vs GPU)
  tools dumplayers -m file.gguf -p "..."     embed+hidden per layer -> .bin/.json
  tools comparelayers -a dump -b dump        compare hidden (or -m CPU vs GPU)
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
	case "dumplayers":
		err = runDumpLayers(os.Args[2:])
	case "comparelayers":
		err = runCompareLayers(os.Args[2:])
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
