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
| `-b` / `--n-batch`                | `32`    | prefill chunk (dense=GEMM; MoE/MLA=serial tok/chunk; see [cli.md](cli.md))          |
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

**DoD engineering (closed 2026-11-12):** dense decode residency (no host<->device between layers); fused batch FFN + MatMulMat `skip_vec` prefill; greedy `logits_argmax`; MoE/MLA/SWA `-b>1`; GPU KV reclaim; lazy host KV; multi-GPU P2P overlap; Q8_1 fused. See [todo.md](../../GoGGUF-docs/todo.md) § A.

**Baseline (A1, RTX 5060 Ti sm_120, 2026-10-28):** `prompt_tokens≈601`, `prefill_tps≈43`, `decode_tps≈15.4` (`-b 32 -n 128 -ngl 999`). **Paired ratio re-verify** against local `llama-bench` (e.g. `/home/.../llama-b10988/llama-bench`) when `/dev/nvidia*` is available - absolute baseline above; Wave-1 PCIe cuts should raise prefill/decode before computing `prefill_tps/pp` and `decode_tps/tg`.

### Metal (macOS) bench protocol

Not run yet: no numbers are recorded here. Run on the Mac you want to measure (Apple Silicon or Intel + Metal GPU), same GGUF / prompt / `-n` / `-b` for every row:

```bash
# GoGGUF (Metal build: cgo + Xcode command line tools)
PROMPT=$(python3 -c 'print("Hello world. "*200)')   # ~600 tokens
go test -tags metal ./pkg/device/metal/             # parity first (GPU tests skip without Metal)
go build -tags metal -o build/tools ./cmd/tools/
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 1  -ngl 999 --runs 3 --json   # loop / decode-shaped baseline
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 32 -ngl 999 --runs 3 --json   # B3 GEMM prefill
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 32 -ngl 0   --runs 3 --json   # CPU reference

# llama.cpp Metal build on the same machine / GGUF / -ngl / -b
./llama-bench -m models/Qwen3-1.7B-Q8_0.gguf -p 512 -n 128 -ngl 999 -b 32
```

Record per run: machine + chip, macOS version, GoGGUF commit, `prefill_tps`, `decode_tps`, `prompt_tokens`, then compute `prefill_tps / pp` and `decode_tps / tg` against `llama-bench`. Also compare greedy tokens vs `-ngl 0` (`tools greedy`) so a faster run is not a wrong one. Only Q8_0 has the GEMM kernel (`matmul_mat_q8_0`, thread-per-output); other quants still loop matvec, so bench them separately and do not compare with the Q8_0 row. Results go into [todo.md](../../GoGGUF-docs/todo.md) § B3.

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
