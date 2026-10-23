# Testing

Integration and golden tests use a smaller model than the examples, so CI and local runs stay lighter on RAM/CPU.

## Test model

Primary golden model: **`Qwen3-1.7B-Q8_0.gguf`**

```bash
mkdir -p models

curl -L -o models/Qwen3-1.7B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
```

Place models under `models/` with fixed names (no test env overrides):

- `models/Qwen3-1.7B-Q8_0.gguf`
- `models/Llama-3.2-1B-Instruct-Q8_0.gguf` (optional)
- `models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf` (optional)

Fixtures: `test/fixtures/`.

## Run tests

```bash
# Unit tests (no GGUF required)
go test ./...

# Integration / golden (needs the 1.7B file under models/)
go test -tags=integration ./test/integration/...
```

CI (`.github/workflows/ci.yml`):

| Job                   | What runs                                                         | Models / notes                                     |
|-----------------------|-------------------------------------------------------------------|----------------------------------------------------|
| `test`                | `go vet` + `go test ./...` + CLI build + `gogguf` usage smoke     | no GGUF; matrix: Linux / Windows / macOS           |
| `cross-build`         | CPU cross-compile CLI                                             | linux/darwin/windows * amd64/arm64                 |
| `integration`         | `./pkg/model/qwen3/` + `-tags=integration ./test/integration/...` | Qwen3-1.7B + Llama-3.2-1B; `.bin` fixtures in repo |
| `integration-mistral` | `-tags=integration -run TestMistral`                              | Mistral-7B (separate job: larger download / RAM)   |
| `cuda`                | `-tags=cuda` device/model packages + CPU<->GPU integration        | Qwen3-1.7B; **skips** without NVIDIA GPU           |

Release (`.github/workflows/release.yml`): on `v*` tags, cross-builds CPU `gogguf`/`tools` for linux/darwin/windows * amd64/arm64 and attaches them to a GitHub Release.

CUDA locally (Linux + NVIDIA) - see [platforms.md](platforms.md):

```bash
CGO_ENABLED=1 go test -tags=cuda ./pkg/device/cuda/...
CGO_ENABLED=1 go test -tags='cuda,integration' ./test/integration/
```

## Other architecture fixtures

Locally (same URLs as CI):

```bash
# Llama 3.2 1B Instruct
curl -L -o models/Llama-3.2-1B-Instruct-Q8_0.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Mistral 7B Instruct v0.2
curl -L -o models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## Regenerating Qwen3 fixtures

```bash
go run ./cmd/tools dumplogits -m models/Qwen3-1.7B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_logits
go run ./cmd/tools dumplogits -m models/Qwen3-1.7B-Q8_0.gguf --chat -p Hello -o test/fixtures/qwen3_chat_hello_logits

go run ./cmd/tools dumplayers -m models/Qwen3-1.7B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_layers
go run ./cmd/tools dumplayers -m models/Qwen3-1.7B-Q8_0.gguf --chat -p Hello -o test/fixtures/qwen3_chat_hello_layers

go run ./cmd/tools layerlogits -m models/Qwen3-1.7B-Q8_0.gguf -p Hello -top 5
go run ./cmd/tools layerlogits -m models/Qwen3-1.7B-Q8_0.gguf --chat -p Hello -top 5

go run ./cmd/tools greedy -m models/Qwen3-1.7B-Q8_0.gguf --chat "Count from 1 to 20, one number per line." -n 50
```

Then update JSON under `test/fixtures/`:

- `qwen3_golden.json` - from dumplogits `top`/`greedy` + greedy `tokens` (keep `encode_hello: [9707]`)
- `qwen3_layers.json` - from dumplayers `embed_rms` / `layer_rms` / `greedy`
- `qwen3_layer_logits.json` - both `cases` from the two `layerlogits` runs (`"model": "Qwen3-1.7B-Q8_0"`)

```bash
go test -tags=integration -count=1 -run 'TestGoldenFixture|TestLayersFixture|TestLayerLogits' ./test/integration/
```
