# Tools

## `debugtok`

Checks prompt encoding and post-prefill logits: top-5 tokens and greedy next.

```bash
go run ./cmd/tools debugtok ./models/Qwen3-4B-Q8_0.gguf "Hello"
```

## `vocab`

Shows Qwen3 config (`head_dim`, head count) and special token IDs in the vocabulary.

```bash
go run ./cmd/tools vocab ./models/Qwen3-4B-Q8_0.gguf
```

## `bench`

Measures inference speed: TTFT, prefill/decode tok/s. Use `--compare` for CPU vs GPU.

```bash
# GPU build / OS support: platforms.md
CGO_ENABLED=1 go build -tags cuda -o build/tools ./cmd/tools

./build/tools bench -m ./models/Qwen3-4B-Q8_0.gguf -p "Hello" -n 128 --chat
./build/tools bench -m model.gguf -p "Hello" -n 64 -ngl 28 --runs 3 --json
./build/tools bench -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -n 32 --compare -c 2048 --runs 2 --warmup 1
```

| Flag                              | Default | Description                                                                         |
|-----------------------------------|---------|-------------------------------------------------------------------------------------|
| `-m`                              | -       | path to `.gguf`                                                                     |
| `-p`                              | `Hello` | prompt                                                                              |
| `-n`                              | `128`   | decode token count                                                                  |
| `-ngl` / `-dev` / `-tensor-split` | -       | GPU - see [platforms.md](platforms.md); with `--compare`, `-ngl 0` means all layers |
| `-b` / `--n-batch`                | `1`     | prefill chunk (same limits as `run`; see [cli.md](cli.md))                          |
| `-c`                              | `0`     | max GPU KV length (0 = auto, up to 4096)                                            |
| `--chat`                          | `false` | chat template                                                                       |
| `--runs`                          | `1`     | runs to average                                                                     |
| `--warmup`                        | `1`     | warmup runs                                                                         |
| `--json`                          | `false` | JSON output                                                                         |
| `--compare`                       | `false` | CPU (`ngl=0`) vs GPU (`-ngl`)                                                       |

`--compare` prints prefill/decode tok/s. VRAM / device lines: [platforms.md](platforms.md).

### vs `llama-bench`

Same GGUF, same `-ngl` / prompt length / decode `-n`, and matching prefill batch (`-b` here vs `-b` / `-n_batch` in llama.cpp). Map:

| GoGGUF `tools bench` | llama.cpp `llama-bench`                  |
|----------------------|------------------------------------------|
| prefill tok/s        | `pp` (prompt processing)                 |
| decode tok/s         | `tg` (token generation)                  |
| TTFT                 | first-token latency (not always printed) |

Example pair (adjust paths / layer count). Use a **long** `-p` (≥512 tokens) so prefill is measurable; a one-token prompt makes `prefill_tps` look like decode:

```bash
# GoGGUF (CUDA build)
PROMPT=$(python3 -c 'print("Hello world. "*200)')   # ~600 tokens
CGO_ENABLED=1 go build -tags cuda -o build/tools ./cmd/tools/
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 32 -ngl 999 --runs 3 --json

# llama.cpp (same machine / GGUF / -ngl / -b)
./llama-bench -m models/Qwen3-1.7B-Q8_0.gguf -p 512 -n 128 -ngl 999 -b 32
```

**DoD targets**: prefill ≥~0.5* llama-bench `pp`, decode ≥~0.7* `tg` at `-b`≥32 on dense fused quants (Qwen3-1.7B-Q8_0 reference).

**Baseline (A1, RTX 5060 Ti sm_120, 2026-10-28):** GoGGUF after F32 `matmul_mat_f32` + residency decode without host k/v mirror - `prompt_tokens≈601`, `prefill_tps≈43`, `decode_tps≈15.4` (`-b 32 -n 128 -ngl 999`). Re-run the pair above against local `llama-bench` to compute ratios; overall DoD remains open until those ratios land.

## `greedy`

Greedy decode N tokens as JSON token IDs (for golden).

```bash
go run ./cmd/tools greedy -m models/Qwen3-4B-Q8_0.gguf --chat "Hello" -n 50
```

## `debuglayers`

Per-layer RMS after embed/each transformer layer plus final top logits.

```bash
go run ./cmd/tools debuglayers -m ./models/Qwen3-4B-Q8_0.gguf -p "Hello"
```

## `layerlogits`

Per-layer greedy next and top-k logits (JSON fixture generator).

```bash
go run ./cmd/tools layerlogits -m models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -top 5
```

## `dumplogits`

Full-vocab prefill logits -> `prefix.bin` + `prefix.json`.

```bash
go run ./cmd/tools dumplogits -m models/Qwen3-4B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_logits
```

## `comparelogits`

Compare two dumps or CPU vs GPU (full vocab).

```bash
go run ./cmd/tools comparelogits -a test/fixtures/qwen3_raw_hello_logits -b test/fixtures/qwen3_raw_hello_logits

CGO_ENABLED=1 go run -tags cuda ./cmd/tools comparelogits -m models/Qwen3-4B-Q8_0.gguf -p Hello -tol 0.01 -align
```

## `dumplayers`

Embed + per-layer hidden states -> `prefix.bin` + `prefix.json`.

```bash
go run ./cmd/tools dumplayers -m models/Qwen3-4B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_layers

go run ./cmd/tools dumplayers -m models/Qwen3-4B-Q8_0.gguf --chat -p Hello -o test/fixtures/qwen3_chat_hello_layers
```

## `comparelayers`

Compare two dumps or CPU vs GPU (per-layer hidden)

```bash
go run ./cmd/tools comparelayers -a test/fixtures/qwen3_raw_hello_layers -b test/fixtures/qwen3_raw_hello_layers

CGO_ENABLED=1 go run -tags cuda ./cmd/tools comparelayers -m models/Qwen3-4B-Q8_0.gguf -p Hello -tol 0.01
```
