# CLI

## `gguf info`

Short model summary: GGUF version, architecture, name, tensor count, weight size, context length.

```bash
./build/gogguf info -m ./models/Qwen3-4B-Q8_0.gguf
```

| Flag | Description          |
|------|----------------------|
| `-m` | path to `.gguf` file |

## `gguf inspect`

Full dump of metadata and tensor list (name, type, dimensions, size in bytes).

```bash
./build/gogguf inspect ./models/Qwen3-4B-Q8_0.gguf
```

Positional argument - file path, no flags.

## `gguf run`

Text generation: prompt prefill -> autoregressive decode -> stdout.

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -n 64
```

Download from Hugging Face (cached under `~/.cache/huggingface/hub`):

```bash
./build/gogguf run -hf Qwen/Qwen3-4B-GGUF:Q8_0 --chat -p "Hello" -n 64
```

| Flag                | Default | Description                                             |
|---------------------|---------|---------------------------------------------------------|
| `-m`                | -       | path to `.gguf` file (mutually exclusive with `-hf`)    |
| `-hf` / `--hf-repo` | -       | Hugging Face `owner/repo[:quant]` (download + cache)    |
| `-p`                | -       | prompt text                                             |
| `-n`                | `128`   | max new tokens                                          |
| `--temp`            | `0`     | sampling temperature (`0` = greedy)                     |
| `--top-k`           | `0`     | top-k (`0` = off)                                       |
| `--top-p`           | `1`     | nucleus sampling (`1` = off)                            |
| `--min-p`           | `0`     | min-p sampling (`0` = off)                              |
| `--repeat-penalty`  | `1`     | repetition penalty (`1` = off)                          |
| `--repeat-last-n`   | `64`    | history window for repeat penalty                       |
| `--seed`            | `0`     | PRNG seed                                               |
| `--chat`            | `false` | wrap prompt in ChatML/Qwen template                     |
| `--thinking`        | `false` | Qwen3 thinking mode (with `--chat`)                     |
| `-i`                | `false` | interactive REPL (stdin)                                |
| `-ngl`              | `0`     | matmul N transformer layers on GPU (CUDA build)         |
| `-b` / `--n-batch`  | `1`     | prefill chunk size (Qwen3 CPU; >1 speeds long prefill)  |
| `-dev`              | empty   | GPU ordinal(s) for offload: `1` or `0,1` (layer split)  |
| `-tensor-split`     | empty   | layer proportions across `-dev` devices, e.g. `0.6,0.4` |

Without `[:quant]`, prefers `Q4_K_M`, then `Q8_0`, else the first model `.gguf`. Gated/private repos: set `HF_TOKEN`.
Alternate Hub mirror: `MODEL_ENDPOINT`.

For **Qwen3 Instruct** use `--chat`, otherwise the model will respond incorrectly.

Sampling example:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -n 64 --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 --seed 42
```

With thinking mode:

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat --thinking -p "Hello" -n 64
```

With GPU offload (all 36 layers of Qwen3-4B, CUDA build):

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -ngl 36
```

Interactive mode (with `--chat`, history is kept across turns; `/clear` resets it):

```bash
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -i
```

## `gguf serve`

HTTP server with an OpenAI-compatible API (see [api.md](api.md)).

Graceful shutdown on `Ctrl+C` (SIGINT/SIGTERM).

```bash
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000
```

Or from Hugging Face:

```bash
./build/gogguf serve -hf Qwen/Qwen3-4B-GGUF:Q8_0 --host 127.0.0.1:8000
```

| Flag                | Default          | Description                                          |
|---------------------|------------------|------------------------------------------------------|
| `-m`                | -                | path to `.gguf` file (mutually exclusive with `-hf`) |
| `-hf` / `--hf-repo` | -                | Hugging Face `owner/repo[:quant]`                    |
| `--host`            | `127.0.0.1:8000` | HTTP listen address                                  |
| `-ngl`              | `0`              | matmul N transformer layers on GPU (CUDA build)      |
| `-b` / `--n-batch`  | `1`              | prefill chunk size (Qwen3 CPU)                       |
| `-dev`              | empty            | GPU ordinal(s) for offload: `1` or `0,1`             |
| `-tensor-split`     | empty            | layer proportions across `-dev` devices              |
| `--api-key`         | empty            | Bearer / X-API-Key auth                              |
| `--rate-limit`      | `0`              | requests per minute per IP (`0` = off)               |

See [HTTP API](api.md) for endpoints.
