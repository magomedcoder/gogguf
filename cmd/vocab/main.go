package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

func main() {
	path := "./models/Qwen3-0.6B-Q8_0.gguf"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	r, _ := gogguf.OpenFile(path)
	cfg, _ := qwen3.ParseConfig(r)
	fmt.Printf("config: head_dim=%d heads=%d kv=%d\n", cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads)

	tokens, _ := r.Metadata.StringArray("tokenizer.ggml.tokens")
	for i, s := range tokens {
		if strings.Contains(s, "im_start") || strings.Contains(s, "im_end") || s == "<|endoftext|>" {
			fmt.Printf("%d: %q\n", i, s)
		}
	}
}
