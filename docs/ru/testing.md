# Тестирование

Integration и golden-тесты используют модель меньше, чем в примерах, чтобы CI и локальный прогон меньше нагружали RAM/CPU.

## Тестовая модель

Основная golden-модель: **`Qwen3-1.7B-Q8_0.gguf`**

```bash
mkdir -p models

curl -L -o models/Qwen3-1.7B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
```

Модели кладите в `models/` с фиксированными именами (без тестовых env):

- `models/Qwen3-1.7B-Q8_0.gguf`
- `models/Llama-3.2-1B-Instruct-Q8_0.gguf` (опционально)
- `models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf` (опционально)

Фикстуры: `test/fixtures/`.

## Запуск тестов

```bash
# Юнит-тесты (GGUF не нужен)
go test ./...

# Integration / golden (нужен файл 1.7B в models/)
go test -tags=integration ./test/integration/...
```

CI (`.github/workflows/ci.yml`):

| Job                   | Что гоняет                                                        | Модели / заметки                                   |
|-----------------------|-------------------------------------------------------------------|----------------------------------------------------|
| `test`                | `go vet` + `go test ./...` + сборка CLI + smoke usage `gogguf`    | без GGUF; matrix: Linux / Windows / macOS          |
| `cross-build`         | CPU cross-compile CLI                                             | linux/darwin/windows * amd64/arm64                 |
| `integration`         | `./pkg/model/qwen3/` + `-tags=integration ./test/integration/...` | Qwen3-1.7B + Llama-3.2-1B; `.bin` фикстуры в репо  |
| `integration-mistral` | `-tags=integration -run TestMistral`                              | Mistral-7B (отдельный job: больший download / RAM) |
| `cuda`                | `-tags=cuda` пакеты device/model + CPU<->GPU integration          | Qwen3-1.7B; **skip** без NVIDIA GPU                |

CUDA локально (Linux + NVIDIA) - см. [platforms.md](platforms.md):

```bash
CGO_ENABLED=1 go test -tags=cuda ./pkg/device/cuda/...
CGO_ENABLED=1 go test -tags='cuda,integration' ./test/integration/
```

## Фикстуры других архитектур

Часть тестов ищет опциональные модели и делает skip, если файла нет:

```bash
# Llama 3.2 1B Instruct
curl -L -o models/Llama-3.2-1B-Instruct-Q8_0.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Mistral 7B Instruct v0.2
curl -L -o models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## Перегенерация Qwen3-фикстур

```bash
go run ./cmd/tools dumplogits -m models/Qwen3-1.7B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_logits
go run ./cmd/tools dumplogits -m models/Qwen3-1.7B-Q8_0.gguf --chat -p Hello -o test/fixtures/qwen3_chat_hello_logits

go run ./cmd/tools dumplayers -m models/Qwen3-1.7B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_layers
go run ./cmd/tools dumplayers -m models/Qwen3-1.7B-Q8_0.gguf --chat -p Hello -o test/fixtures/qwen3_chat_hello_layers

go run ./cmd/tools layerlogits -m models/Qwen3-1.7B-Q8_0.gguf -p Hello -top 5
go run ./cmd/tools layerlogits -m models/Qwen3-1.7B-Q8_0.gguf --chat -p Hello -top 5

go run ./cmd/tools greedy -m models/Qwen3-1.7B-Q8_0.gguf --chat "Count from 1 to 20, one number per line." -n 50
```

Дальше обновите JSON в `test/fixtures/`:

- `qwen3_golden.json` - из dumplogits (`top`/`greedy`) + greedy (`tokens`); оставьте `encode_hello: [9707]`
- `qwen3_layers.json` - из dumplayers (`embed_rms` / `layer_rms` / `greedy`)
- `qwen3_layer_logits.json` - оба `cases` из двух запусков `layerlogits` (`"model": "Qwen3-1.7B-Q8_0"`)

```bash
go test -tags=integration -count=1 -run 'TestGoldenFixture|TestLayersFixture|TestLayerLogits' ./test/integration/
```
