# CLI

Examples use a Unix shell. On Windows the binary is `build\gogguf.exe` and the same flags apply in PowerShell (`.\build\gogguf.exe info -m .\models\....gguf`). macOS matches the Unix examples. GPU / OS: [platforms.md](platforms.md).

## `gogguf info`

Short model summary: GGUF version, architecture, name, tensor count, weight size, context length.

```bash
./build/gogguf info -m ./models/Qwen3-4B-Q8_0.gguf
```

| Flag | Description          |
|------|----------------------|
| `-m` | path to `.gguf` file |

## `gogguf inspect`

Full dump of metadata and tensor list (name, type, dimensions, size in bytes).

```bash
./build/gogguf inspect ./models/Qwen3-4B-Q8_0.gguf
```

Positional argument - file path, no flags.

## `gogguf run`

Text generation: prompt prefill -> autoregressive decode -> stdout.

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -n 64
```

Download from Hugging Face (cached under `~/.cache/huggingface/hub`):

```bash
./build/gogguf run -hf Qwen/Qwen3-4B-GGUF:Q8_0 --chat -p "Hello" -n 64
```

| Flag                | Default | Description                                                                              |
|---------------------|---------|------------------------------------------------------------------------------------------|
| `-m`                | -       | path to `.gguf` file (mutually exclusive with `-hf`)                                     |
| `-hf` / `--hf-repo` | -       | Hugging Face `owner/repo[:quant]` (download + cache)                                     |
| `-p`                | -       | prompt text                                                                              |
| `-n`                | `128`   | max new tokens                                                                           |
| `--temp`            | `0`     | sampling temperature (`0` = greedy)                                                      |
| `--top-k`           | `0`     | top-k (`0` = off)                                                                        |
| `--top-p`           | `1`     | nucleus sampling (`1` = off)                                                             |
| `--min-p`           | `0`     | min-p sampling (`0` = off)                                                               |
| `--repeat-penalty`  | `1`     | repetition penalty (`1` = off)                                                           |
| `--repeat-last-n`   | `64`    | history window for repeat penalty                                                        |
| `--seed`            | `0`     | PRNG seed                                                                                |
| `--chat`            | `false` | wrap prompt in ChatML/Qwen template                                                      |
| `--thinking`        | `false` | Qwen3 thinking mode (with `--chat`)                                                      |
| `-i`                | `false` | interactive REPL (stdin)                                                                 |
| `-ngl`              | `0`     | GPU layers (`0` = CPU; needs `-tags cuda` build + `N > 0`): [platforms.md](platforms.md) |
| `-b` / `--n-batch`  | `32`    | prefill chunk; dense arches use batch/GEMM; MoE/MLA/SWA serial (`1`)                     |
| `-dev`              | empty   | GPU device(s) - see [platforms.md](platforms.md)                                         |
| `-tensor-split`     | empty   | multi-GPU layer split - see [platforms.md](platforms.md)                                 |

Without `[:quant]`, prefers `Q4_K_M`, then `Q8_0`, else the first model `.gguf`.

For **Qwen3 Instruct** use `--chat`, otherwise the model will respond incorrectly.

Sampling example:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -n 64 --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 --seed 42
```

With thinking mode:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat --thinking -p "Hello" -n 64
```

GPU (`-ngl`, `-dev`, Docker): [platforms.md](platforms.md).

Interactive mode (with `--chat`, history is kept across turns; `/clear` resets it):

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -i
```

## `gogguf serve`

HTTP server with an OpenAI-compatible API (see [api.md](api.md)).

Graceful shutdown on `Ctrl+C` (SIGINT/SIGTERM).

```bash
# CPU (default: -ngl 0)
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000

# GPU: CUDA build + -ngl > 0 (build alone does not enable offload)
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000 -ngl 999
```

Or from Hugging Face:

```bash
./build/gogguf serve -hf Qwen/Qwen3-4B-GGUF:Q8_0 --host 127.0.0.1:8000 -ngl 999
```

| Flag                     | Default          | Description                                                                         |
|--------------------------|------------------|-------------------------------------------------------------------------------------|
| `-m`                     | -                | path to `.gguf` file (mutually exclusive with `-hf`)                                |
| `-hf` / `--hf-repo`      | -                | Hugging Face `owner/repo[:quant]`                                                   |
| `--host`                 | `127.0.0.1:8000` | HTTP listen address                                                                 |
| `-ngl`                   | `0`              | GPU layers (`0` = CPU); needs `-tags cuda` + `N > 0` - [platforms.md](platforms.md) |
| `-dev` / `-tensor-split` | empty            | multi-GPU - see [platforms.md](platforms.md)                                        |
| `-b` / `--n-batch`       | `32`             | prefill chunk (dense batch/GEMM; MoE/MLA/SWA serial)                                |
| `--api-key`              | empty            | Bearer / X-API-Key auth                                                             |
| `--rate-limit`           | `0`              | requests per minute per IP (`0` = off)                                              |

See [HTTP API](api.md) for endpoints.

## Environment (Hugging Face)

Used by `-hf` / `pkg/hf` only. Inference and residency are **not** controlled by env - use `LoadOptions` ([library.md](library.md)).

| Variable                | Description                                               |
|-------------------------|-----------------------------------------------------------|
| `HF_TOKEN`              | Auth for gated/private Hub repos.                         |
| `MODEL_ENDPOINT`        | Hub base URL override (default `https://huggingface.co`). |
| `LLAMA_CACHE`           | Model cache directory (highest priority).                 |
| `HF_HUB_CACHE`          | Hub cache directory.                                      |
| `HUGGINGFACE_HUB_CACHE` | Alias of Hub cache.                                       |
| `HF_HOME`               | Base dir; cache is `$HF_HOME/hub`.                        |
| `XDG_CACHE_HOME`        | Cache is `$XDG_CACHE_HOME/huggingface/hub`.               |
| `LOCALAPPDATA`          | Windows fallback: `%LOCALAPPDATA%\huggingface\hub`.       |

Cache resolution order: `LLAMA_CACHE` -> `HF_HUB_CACHE` -> `HUGGINGFACE_HUB_CACHE` -> `HF_HOME/hub` -> `XDG_CACHE_HOME/huggingface/hub` -> (Windows) `LOCALAPPDATA` -> `~/.cache/huggingface/hub`.

`CUDA_VISIBLE_DEVICES` is the standard CUDA driver filter (not GoGGUF-specific); it affects which GPUs the process sees when `-ngl > 0` - see [platforms.md](platforms.md).
