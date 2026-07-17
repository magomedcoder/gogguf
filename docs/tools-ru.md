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

Замер скорости inference: TTFT, prefill/decode tok/s.

```bash
go build -o build/tools ./cmd/tools

./build/tools bench -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Привет" -n 128 --chat

./build/tools bench -m model.gguf -p "Привет" -n 64 -ngl 28 --runs 3 --json
```

| Флаг       | По умолчанию | Описание                |
|------------|--------------|-------------------------|
| `-m`       | -            | путь к `.gguf`          |
| `-p`       | `Привет`     | промпт                  |
| `-n`       | `128`        | число decode-токенов    |
| `-ngl`     | `0`          | GPU offload (CUDA)      |
| `--chat`   | `false`      | chat template           |
| `--runs`   | `1`          | прогонов для усреднения |
| `--warmup` | `1`          | прогревочных прогонов   |
| `--json`   | `false`      | вывод в JSON            |

## `greedy`

Greedy decode N токенов в JSON (token IDs) для сверки с golden.

```bash
go run ./cmd/tools greedy -m models/Qwen3-0.6B-Q8_0.gguf --chat "Hello" -n 50
```

## `debuglayers`

Послойный RMS после embed/каждого слоя + итоговые top logits.

```bash
go run ./cmd/tools debuglayers -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Привет"
```

## `layerlogits`

Greedy next и top-k logits по слоям (генератор JSON fixture).

```bash
go run ./cmd/tools layerlogits -m models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -top 5
```
