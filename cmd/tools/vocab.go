package main

import (
	"fmt"
	"strings"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/model/qwen3"
)

// runVocab показывает конфиг модели и ID special tokens
func runVocab(args []string) error {
	path := "./models/Qwen3-0.6B-Q8_0.gguf"
	if len(args) > 0 {
		path = args[0]
	}

	r, err := gogguf.OpenFile(path)
	if err != nil {
		return err
	}

	cfg, err := qwen3.ParseConfig(r)
	if err != nil {
		return err
	}

	fmt.Printf("config: head_dim=%d heads=%d kv=%d\n", cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads)

	tokens, err := r.Metadata.StringArray("tokenizer.ggml.tokens")
	if err != nil {
		return err
	}

	for i, s := range tokens {
		if strings.Contains(s, "im_start") || strings.Contains(s, "im_end") || s == "<|endoftext|>" {
			fmt.Printf("%d: %q\n", i, s)
		}
	}
	return nil
}
