# Утилиты

[English version](tools.md)

## `debugtok`

Проверяет encode промпта и logits после prefill: top-5 токенов и greedy-следующий.

```bash
go run ./cmd/debugtok ./models/Qwen3-0.6B-Q8_0.gguf "Привет"
```

## `vocab`

Показывает конфиг Qwen3 (`head_dim`, число heads) и id special tokens в словаре.

```bash
go run ./cmd/vocab ./models/Qwen3-0.6B-Q8_0.gguf
```

## `bench`

Замер скорости inference: TTFT, prefill/decode tok/s.

```bash
go build -o build/bench ./cmd/bench

./build/bench -m ./models/Qwen3-0.6B-Q8_0.gguf -p "Привет" -n 128 --chat

./build/bench -m model.gguf -p "Привет" -n 64 -ngl 28 --runs 3 --json
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
