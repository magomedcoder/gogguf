# GoGGUF

[Русская версия](README-ru.md)

**GoGGUF** is a lightweight way to run GGUF models in Go without llama.cpp.

> **Actively developed.** GPU offload (CUDA) is still early and evolving.

Use it as a **library** (inference from Go code) or as an **HTTP server** (`gguf serve` or the `server` package).

**No external Go dependencies** - standard library only.

Optional: **CUDA** via Driver API (`libcuda.so`, build with `-tags cuda`, CGO).

## What works today

- GGUF v2/v3 parsing (`info`, `inspect`), memory-map (`LoadMapped`, zero-copy `RawView`);
- dequantization and matmul: Q8_0, Q4_0, Q4_K;
- basic ops: RoPE, RMSNorm, GQA attention, SwiGLU;
- **SIMD** FP32 matmul: AVX2 (amd64), NEON (arm64); Q8_0 dot: AVX2 (amd64);
- **Qwen3** and **Llama 3** forward pass + KV-cache;
- BPE tokenizer from GGUF metadata;
- ChatML/Qwen and Jinja chat templates (`--chat`, `--thinking`, `FormatChatUser`);
- text generation: `gguf run` (greedy / temperature / top-k / top-p / min-p / repeat penalty);
- HTTP server: `gguf serve` (`/v1/models`, `/v1/chat/completions`, JSON + SSE);
- **CUDA offload** (`-ngl N`): matmul for the first N transformer layers on GPU (build with `-tags cuda`).

## Models

**Works:** Qwen3, Llama 3
**Soon:** Llama 2, Mistral, Phi, Gemma

Details: [docs/models.md](docs/models.md).

## Test model

For development and testing we use Qwen3-0.6B-Q8_0.gguf:

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

## Quick start

```bash
go build -o build/gogguf ./cmd/gogguf

./build/gogguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -n 64
```

## Documentation

| Section                                    | Description                            |
|--------------------------------------------|----------------------------------------|
| [docs/build.md](docs/build.md)             | Build: CPU, CUDA, Docker               |
| [docs/cli.md](docs/cli.md)                 | CLI: `info`, `inspect`, `run`, `serve` |
| [docs/api.md](docs/api.md)                 | HTTP server API                        |
| [docs/library.md](docs/library.md)         | Inference from Go                      |
| [docs/tools.md](docs/tools.md)             | `debugtok`, `vocab`, `bench`           |
| [docs/GGUF-FORMAT.md](docs/GGUF-FORMAT.md) | GGUF format spec                       |
| [docs/models.md](docs/models.md)           | Supported and planned models           |
