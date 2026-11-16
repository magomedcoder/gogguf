# Утилиты

## `debugtok`

Проверяет encode промпта и logits после prefill: top-5 токенов и greedy-следующий.

```bash
go run ./cmd/tools debugtok ./models/Qwen3-4B-Q8_0.gguf "Hello"
```

## `vocab`

Показывает конфиг Qwen3 (`head_dim`, число heads) и id special tokens в словаре.

```bash
go run ./cmd/tools vocab ./models/Qwen3-4B-Q8_0.gguf
```

## `bench`

Замер скорости inference: TTFT, prefill/decode tok/s. Режим `--compare` - CPU vs GPU.

```bash
# сборка GPU / ОС: platforms.md
CGO_ENABLED=1 go build -tags cuda -o build/tools ./cmd/tools

./build/tools bench -m ./models/Qwen3-4B-Q8_0.gguf -p "Hello" -n 128 --chat
./build/tools bench -m model.gguf -p "Hello" -n 64 -ngl 28 --runs 3 --json
./build/tools bench -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -n 32 --compare -c 2048 --runs 2 --warmup 1
```

| Флаг                              | По умолчанию | Описание                                                                    |
|-----------------------------------|--------------|-----------------------------------------------------------------------------|
| `-m`                              | -            | путь к `.gguf`                                                              |
| `-p`                              | `Hello`      | промпт                                                                      |
| `-n`                              | `128`        | число decode-токенов                                                        |
| `-ngl` / `-dev` / `-tensor-split` | -            | GPU - см. [platforms.md](platforms.md); при `--compare` `-ngl 0` = все слои |
| `-b` / `--n-batch`                | `32`         | chunk prefill (dense=GEMM; MoE/MLA=serial tok; см. [cli.md](cli.md))        |
| `-c`                              | `0`          | макс. длина GPU KV (`0` = авто, до 4096)                                    |
| `--chat`                          | `false`      | chat template                                                               |
| `--runs`                          | `1`          | прогонов для усреднения                                                     |
| `--warmup`                        | `1`          | прогревочных прогонов                                                       |
| `--json`                          | `false`      | вывод в JSON                                                                |
| `--compare`                       | `false`      | CPU (`ngl=0`) vs GPU (`-ngl`)                                               |

`--compare` печатает prefill/decode tok/s. Устройство / VRAM: [platforms.md](platforms.md).

### vs `llama-bench`

Тот же GGUF, те же `-ngl` / длина промпта / decode `-n` и тот же prefill batch (`-b` здесь и `-b` / `-n_batch` в llama.cpp).

| GoGGUF `tools bench` | llama.cpp `llama-bench`                        |
|----------------------|------------------------------------------------|
| prefill tok/s        | `pp` (prompt processing)                       |
| decode tok/s         | `tg` (token generation)                        |
| TTFT                 | задержка первого токена (не всегда печатается) |

Для measurable prefill нужен **длинный** `-p` (≥512 токенов); один токен делает `prefill_tps` похожим на decode:

```bash
PROMPT=$(python3 -c 'print("Hello world. "*200)')
CGO_ENABLED=1 go build -tags cuda -o build/tools ./cmd/tools/
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 32 -ngl 999 --runs 3 --json
./llama-bench -m models/Qwen3-1.7B-Q8_0.gguf -p 512 -n 128 -ngl 999 -b 32
```

**DoD**: prefill ≥~0.5* `pp`, decode ≥~0.7* `tg` при `-b`≥32 на dense fused (эталон Qwen3-1.7B-Q8_0).

**DoD engineering (закрыт 2026-11-12):** dense decode residency; fused batch FFN + MatMulMat `skip_vec`; greedy `logits_argmax`; MoE/MLA/SWA `-b>1`; GPU KV reclaim; lazy host KV; multi-GPU P2P overlap; Q8_1 fused. Статус GPU: [platforms.md](platforms.md).

**Baseline (A1, RTX 5060 Ti sm_120, 2026-10-28):** `prompt_tokens≈601`, `prefill_tps≈43`, `decode_tps≈15.4`. **Paired ratio re-verify** с локальным `llama-bench`, когда есть `/dev/nvidia*`.

### Протокол bench на Metal (macOS)

Пока не прогонялся: чисел здесь нет. Запускать на том Mac, который измеряем (Apple Silicon или Intel + Metal GPU); для всех строк один GGUF / промпт / `-n` / `-b`:

```bash
# GoGGUF (Metal-сборка: cgo + Xcode command line tools)
PROMPT=$(python3 -c 'print("Hello world. "*200)')   # ~600 токенов
go test -tags metal ./pkg/device/metal/             # сначала parity (GPU-тесты skip без Metal)
go build -tags metal -o build/tools ./cmd/tools/
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 1  -ngl 999 --runs 3 --json   # baseline без GEMM
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 32 -ngl 999 --runs 3 --json   # B3 GEMM prefill
./build/tools bench -m models/Qwen3-1.7B-Q8_0.gguf -p "$PROMPT" -n 128 -b 32 -ngl 0   --runs 3 --json   # эталон CPU

# llama.cpp (Metal-сборка) на той же машине / GGUF / -ngl / -b
./llama-bench -m models/Qwen3-1.7B-Q8_0.gguf -p 512 -n 128 -ngl 999 -b 32
```

Фиксировать для каждого прогона: машина и чип, версия macOS, коммит GoGGUF, `prefill_tps`, `decode_tps`, `prompt_tokens`; затем считать `prefill_tps / pp` и `decode_tps / tg` относительно `llama-bench`. Дополнительно сверять greedy-токены с `-ngl 0` (`tools greedy`), чтобы быстрый прогон не был неверным. GEMM-kernel (`matmul_mat_q8_0`, thread-per-output) есть только у Q8_0; остальные quant по-прежнему крутят matvec - бенчить их отдельно и не сравнивать со строкой Q8_0. Измеренные числа вписать в этот раздел, когда будут.

## `greedy`

Greedy decode N токенов в JSON (token IDs) для сверки с golden.

```bash
go run ./cmd/tools greedy -m models/Qwen3-4B-Q8_0.gguf --chat "Hello" -n 50
```

## `debuglayers`

Послойный RMS после embed/каждого слоя + итоговые top logits.

```bash
go run ./cmd/tools debuglayers -m ./models/Qwen3-4B-Q8_0.gguf -p "Hello"
```

## `layerlogits`

Greedy next и top-k logits по слоям (генератор JSON fixture).

```bash
go run ./cmd/tools layerlogits -m models/Qwen3-4B-Q8_0.gguf --chat -p "Hello" -top 5
```

## `dumplogits`

Полный vocab logits после prefill -> `prefix.bin` + `prefix.json`.

```bash
go run ./cmd/tools dumplogits -m models/Qwen3-4B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_logits
```

## `comparelogits`

Сверка двух dump или CPU vs GPU (полный vocab).

```bash
go run ./cmd/tools comparelogits -a test/fixtures/qwen3_raw_hello_logits -b test/fixtures/qwen3_raw_hello_logits

CGO_ENABLED=1 go run -tags cuda ./cmd/tools comparelogits -m models/Qwen3-4B-Q8_0.gguf -p Hello -tol 0.01 -align
```

## `dumplayers`

Embed + hidden после каждого слоя -> `prefix.bin` + `prefix.json`.

```bash
go run ./cmd/tools dumplayers -m models/Qwen3-4B-Q8_0.gguf -p Hello -o test/fixtures/qwen3_raw_hello_layers

go run ./cmd/tools dumplayers -m models/Qwen3-4B-Q8_0.gguf --chat -p Hello -o test/fixtures/qwen3_chat_hello_layers
```

## `comparelayers`

Сверка двух dump или CPU vs GPU (hidden по слоям).

```bash
go run ./cmd/tools comparelayers -a test/fixtures/qwen3_raw_hello_layers -b test/fixtures/qwen3_raw_hello_layers

CGO_ENABLED=1 go run -tags cuda ./cmd/tools comparelayers -m models/Qwen3-4B-Q8_0.gguf -p Hello -tol 0.01
```
