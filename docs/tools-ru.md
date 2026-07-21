# Утилиты

[English version](tools.md)

## `debugtok`

Проверяет encode промпта и logits после prefill: top-5 токенов и greedy-следующий.

```bash
go run ./cmd/tools debugtok ./models/Qwen3-0.6B-Q8_0.gguf "Привет"
```

## `vocab`

Показывает конфиг Qwen3 (`head_dim`, число heads) и id special tokens в словаре.

```bash
go run ./cmd/tools vocab ./models/Qwen3-0.6B-Q8_0.gguf
```

## `bench`

Замер скорости inference: TTFT, prefill/decode tok/s. Режим `--compare` - CPU vs GPU.

```bash
# сборка с CUDA
CGO_ENABLED=1 go build -tags cuda -o build/tools ./cmd/tools

./build/tools bench -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Привет" -n 128 --chat

./build/tools bench -m model.gguf -p "Привет" -n 64 -ngl 28 --runs 3 --json

./build/tools bench -m ./models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -n 32 --compare -c 2048 --runs 2 --warmup 1
```

| Флаг        | По умолчанию | Описание                                         |
|-------------|--------------|--------------------------------------------------|
| `-m`        | -            | путь к `.gguf`                                   |
| `-p`        | `Привет`     | промпт                                           |
| `-n`        | `128`        | число decode-токенов                             |
| `-ngl`      | `0`          | GPU offload (CUDA); при `--compare` 0 = все слои |
| `-c`        | `0`          | макс. длина GPU KV (0 = авто, до 4096)           |
| `--chat`    | `false`      | chat template                                    |
| `--runs`    | `1`          | прогонов для усреднения                          |
| `--warmup`  | `1`          | прогревочных прогонов                            |
| `--json`    | `false`      | вывод в JSON                                     |
| `--compare` | `false`      | CPU (`ngl=0`) vs GPU (`-ngl`)                    |

`--compare` печатает таблицу prefill/decode tok/s и флаг `GPU decode быстрее CPU`.

## `greedy`

Greedy decode N токенов в JSON (token IDs) для сверки с golden.

```bash
go run ./cmd/tools greedy -m models/Qwen3-0.6B-Q8_0.gguf --chat "Привет" -n 50
```

## `debuglayers`

Послойный RMS после embed/каждого слоя + итоговые top logits.

```bash
go run ./cmd/tools debuglayers -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Привет"
```

## `layerlogits`

Greedy next и top-k logits по слоям (генератор JSON fixture).

```bash
go run ./cmd/tools layerlogits -m models/Qwen3-0.6B-Q8_0.gguf --chat -p "Привет" -top 5
```

## `dumplogits`

Полный vocab logits после prefill -> `prefix.bin` + `prefix.json`.

```bash
go run ./cmd/tools dumplogits -m models/Qwen3-0.6B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_logits

go run ./cmd/tools dumplogits -m models/Qwen3-0.6B-Q8_0.gguf --chat -p Hello -o test/fixtures/qwen3_chat_hello_logits
```

## `comparelogits`

Сверка двух dump или CPU vs GPU (полный vocab).

```bash
go run ./cmd/tools comparelogits -a test/fixtures/qwen3_raw_hello_logits -b test/fixtures/qwen3_raw_hello_logits

CGO_ENABLED=1 go run -tags cuda ./cmd/tools comparelogits -m models/Qwen3-0.6B-Q8_0.gguf -p Hello -tol 0.01 -align
```
