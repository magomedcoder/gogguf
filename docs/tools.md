# Tools

[Русская версия](tools-ru.md)

## `debugtok`

Checks prompt encoding and post-prefill logits: top-5 tokens and greedy next.

```bash
go run ./cmd/debugtok ./models/Qwen3-0.6B-Q8_0.gguf "Hello"
```

## `vocab`

Shows Qwen3 config (`head_dim`, head count) and special token IDs in the vocabulary.

```bash
go run ./cmd/vocab ./models/Qwen3-0.6B-Q8_0.gguf
```

## `bench`

Measures inference speed: TTFT, prefill/decode tok/s.

```bash
go build -o build/bench ./cmd/bench

./build/bench -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Hello" -n 128 --chat

./build/bench -m model.gguf -p "Hello" -n 64 -ngl 28 --runs 3 --json
```

| Flag       | Default | Description        |
|------------|---------|--------------------|
| `-m`       | -       | path to `.gguf`    |
| `-p`       | `Hello` | prompt             |
| `-n`       | `128`   | decode token count |
| `-ngl`     | `0`     | GPU offload (CUDA) |
| `--chat`   | `false` | chat template      |
| `--runs`   | `1`     | runs to average    |
| `--warmup` | `1`     | warmup runs        |
| `--json`   | `false` | JSON output        |
