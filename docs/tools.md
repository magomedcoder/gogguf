# Tools

[Русская версия](tools-ru.md)

## `debugtok`

Checks prompt encoding and post-prefill logits: top-5 tokens and greedy next.

```bash
go run ./cmd/tools debugtok ./models/Qwen3-0.6B-Q8_0.gguf "Hello"
```

## `vocab`

Shows Qwen3 config (`head_dim`, head count) and special token IDs in the vocabulary.

```bash
go run ./cmd/tools vocab ./models/Qwen3-0.6B-Q8_0.gguf
```

## `bench`

Measures inference speed: TTFT, prefill/decode tok/s.

```bash
go build -o build/tools ./cmd/tools

./build/tools bench -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Hello" -n 128 --chat

./build/tools bench -m model.gguf -p "Hello" -n 64 -ngl 28 --runs 3 --json
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

## `greedy`

Greedy decode N tokens as JSON token IDs (for golden).

```bash
go run ./cmd/tools greedy -m models/Qwen3-0.6B-Q8_0.gguf --chat "Hello" -n 50
```

## `debuglayers`

Per-layer RMS after embed/each transformer layer plus final top logits.

```bash
go run ./cmd/tools debuglayers -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Hello"
```

## `layerlogits`

Per-layer greedy next and top-k logits (JSON fixture generator).

```bash
go run ./cmd/tools layerlogits -m models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -top 5
```
