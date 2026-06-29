# CLI

[Русская версия](cli-ru.md)

## `gguf info`

Short model summary: GGUF version, architecture, name, tensor count, weight size, context length.

```bash
./build/gguf info -m ./models/Qwen3-0.6B-Q8_0.gguf
```

| Flag | Description          |
|------|----------------------|
| `-m` | path to `.gguf` file |

## `gguf inspect`

Full dump of metadata and tensor list (name, type, dimensions, size in bytes).

```bash
./build/gguf inspect ./models/Qwen3-0.6B-Q8_0.gguf
```

Positional argument - file path, no flags.

## `gguf run`

Text generation: prompt prefill -> autoregressive decode -> stdout.

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -n 64
```

| Flag               | Default | Description                                     |
|--------------------|---------|-------------------------------------------------|
| `-m`               | -       | path to `.gguf` file                            |
| `-p`               | -       | prompt text                                     |
| `-n`               | `128`   | max new tokens                                  |
| `--temp`           | `0`     | sampling temperature (`0` = greedy)             |
| `--top-k`          | `0`     | top-k (`0` = off)                               |
| `--top-p`          | `1`     | nucleus sampling (`1` = off)                    |
| `--min-p`          | `0`     | min-p sampling (`0` = off)                      |
| `--repeat-penalty` | `1`     | repetition penalty (`1` = off)                  |
| `--repeat-last-n`  | `64`    | history window for repeat penalty               |
| `--seed`           | `0`     | PRNG seed                                       |
| `--chat`           | `false` | wrap prompt in ChatML/Qwen template             |
| `--thinking`       | `false` | Qwen3 thinking mode (with `--chat`)             |
| `-i`               | `false` | interactive REPL (stdin)                        |
| `-ngl`             | `0`     | matmul N transformer layers on GPU (CUDA build) |

For **Qwen3 Instruct** use `--chat`, otherwise the model will respond incorrectly.

Sampling example:

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -n 64 --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 --seed 42
```

With thinking mode:

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat --thinking -p "Hello" -n 64
```

With GPU offload (28 layers, CUDA build):

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -ngl 28
```

Interactive mode (with `--chat`, history is kept across turns; `/clear` resets it):

```bash
./build/gguf run -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -i
```

## `gguf serve`

HTTP server for text generation API.

Graceful shutdown on `Ctrl+C` (SIGINT/SIGTERM).

```bash
./build/gguf serve -m ./models/Qwen3-0.6B-Q8_0.gguf --addr 127.0.0.1:8000
```

| Flag     | Default          | Description                                     |
|----------|------------------|-------------------------------------------------|
| `-m`     | -                | path to `.gguf` file                            |
| `--addr` | `127.0.0.1:8000` | HTTP listen address                             |
| `-ngl`   | `0`              | matmul N transformer layers on GPU (CUDA build) |

See [HTTP API](api.md) for endpoints.
