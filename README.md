# GoGGUF

**GoGGUF** is a lightweight way to run GGUF models in Go without llama.cpp.

> **Actively developed.** GPU offload (CUDA) is still early and evolving.

Use it as a **library** (inference from Go code) or as an **HTTP server** (`gguf serve` or the `server` package).

**No external Go dependencies** - standard library only.

Optional: **CUDA** via Driver API (`libcuda.so`, build with `-tags cuda`, CGO).

## What works today

- GGUF v2/v3 parsing (`info`, `inspect`), memory-map (`LoadMapped`, zero-copy `RawView`);
- dequantization and matmul: Q8_0, Q4_0, Q4_K, Q5_K, Q6_K;
- basic ops: RoPE, RMSNorm, GQA attention, SwiGLU;
- **SIMD** FP32 matmul: AVX2 (amd64), NEON (arm64); Q8_0 dot: AVX2 (amd64);
- **Qwen3**, **Llama 2/3**, **Mistral**, **Phi-2/3**, **Gemma/Gemma2** forward + KV-cache;
- BPE tokenizer from GGUF metadata;
- ChatML/Qwen and Jinja chat templates (`--chat`, `--thinking`, `FormatChatUser`);
- text generation: `gguf run` (greedy / temperature / top-k / top-p / min-p / repeat penalty);
- HTTP server: `gguf serve` (`/v1/models`, `/v1/chat/completions`, JSON + SSE);
- **CUDA offload** (`-ngl N`): matmul for the first N transformer layers on GPU (build with `-tags cuda`).

## Models

**Works:** Qwen3, Llama 2/3, Mistral, Phi-2/3, Gemma/Gemma2
**Soon:** Vision, dedicated embedding models

Details: [docs/models.md](docs/models.md).

## Quick start

Example model for running the CLI: `Qwen3-4B-Q8_0.gguf`

```bash
# download model
mkdir -p models
curl -L -o models/Qwen3-4B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf

# build CLI
go build -o build/gogguf ./cmd/gogguf

# run chat in the terminal
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -n 64

# or start the HTTP API (OpenAI-compatible)
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000
```

Optional simple chat UI ([web-ui](web-ui/)): keep `serve` running, then in another terminal:

```bash
cd web-ui
yarn install # or npm install
yarn dev # http://localhost:5173 - proxies /api to :8000
```

API details: [docs/api.md](docs/api.md). UI details: [web-ui/README.md](web-ui/README.md).

## Documentation

* [Build: CPU, CUDA, Docker](docs/build.md)
* [CLI: `info`, `inspect`, `run`, `serve`](docs/cli.md)
* [HTTP server API](docs/api.md)
* [Simple web UI](web-ui/README.md)
* [Inference from Go](docs/library.md)
* [`debugtok`, `vocab`, `bench`](docs/tools.md)
* [GGUF format spec](docs/GGUF-FORMAT.md)
* [Supported and planned models](docs/models.md)
* [Testing](docs/testing.md)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

---

[Русская версия](README-ru.md)
