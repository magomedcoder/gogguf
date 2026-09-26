# Сборка

[English version](build.md)

## Локально

```bash
go build -o build/gogguf ./cmd/gogguf
```

Без CGO - кросс-компиляция на любую платформу.

## CUDA (NVIDIA GPU, опционально)

Требуется драйвер NVIDIA (`libcuda.so`) и CGO.

CUDA Toolkit не нужен - используется Driver API через `dlopen`.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf
```

Проверка GPU matmul:

```bash
CGO_ENABLED=1 go test -tags=cuda ./pkg/gpu/cuda/...
```

`-ngl N` - offload первых N transformer-слоёв на GPU (макс. `block_count`; Qwen3-0.6B - 28).

На GPU: matmul (FP32, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K) с CUDA Graphs (полный HtoD+kernel+DtoH или kernel-only, если vec уже на GPU), layer-graph для FFN, attn+FFN residual, QKV+RoPE и attention (ключ `seq_len`; KV append отдельно), FFN residency, QKV+RoPE+attn residency, attn+FFN residual (WO+RMSNorm+FFN, parallel RMSNorm apply), attention (+ softmax), KV-cache. Отключить: `GGUF_QKV_RESIDENCY=0` / `GGUF_ATTN_FFN_RESIDENCY=0`. Освобождать VRAM через `Engine.Close()`.

Blackwell (sm_120, RTX 50xx): нужен PTX 8.7+; Q8_0 scale конвертируется в FP32 при загрузке на GPU (без PTX f16).

Без `-tags cuda` при `-ngl > 0` будет ошибка `gpu: CUDA недоступна`.

### Что реально считается на GPU

Для **Qwen3 dense** при `-ngl` = все слои (веса Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K) residual stream живёт на устройстве:

| Этап                                                       | Где                      | Примечание                                     |
|------------------------------------------------------------|--------------------------|------------------------------------------------|
| tokenizer, chat template                                   | CPU                      | -                                              |
| embedding токена                                           | CPU + один HtoD на токен | `HiddenUpload` пишет в `d_resid`               |
| `attn_norm` + QKV + QK-norm + RoPE + KV append + attention | GPU                      | fused `QKVRoPEAttention*Cached`, CUDA Graph    |
| WO + residual + `ffn_norm` + FFN (SwiGLU) + residual       | GPU                      | fused `AttnFFNResidual*Cached`, CUDA Graph     |
| residual `x` между слоями                                  | GPU (`d_resid`)          | PCIe на слой нет                               |
| `output_norm` + `lm_head`                                  | GPU                      | `LogitsFromDevice`, один DtoH на `vocab` float |
| sampling, зеркало KV, HTTP                                 | CPU                      | `attn`/`k`/`v` слоя пока уходят DtoH           |

Частично на GPU: prefill при `-b > 1` (matmul чанка - CPU-GEMM, K/V зеркалятся в GPU KV), другие архитектуры (Llama / Mistral: matmul + FFN + attention, без полной residency), MoE-эксперты (CPU), DeepSeek2 MLA (только CPU).
Типы весов без нативных kernels (Q4_1, Q5_0/1, Q2_K, Q3_K, Q8_K) идут по FP32-пути и отключают residency.

### Выбор устройства и multi-GPU

```bash
./build/gogguf run   -m model.gguf -ngl 28 -dev 1 -p "Привет" # вторая видимая карта
./build/gogguf serve -m model.gguf -ngl 28 -dev 0,1 # split слоёв по двум GPU
./build/tools bench  -m model.gguf -ngl 28 -dev 0,1 -tensor-split 0.75,0.25
CUDA_VISIBLE_DEVICES=1 ./build/gogguf run -m model.gguf -ngl 28 -p "Привет"
```

`-dev` принимает ordinal **после** фильтра `CUDA_VISIBLE_DEVICES` (`-dev 0` - первое видимое устройство), выход за диапазон отсекается при инициализации. `run` / `serve` / `tools bench` пишут выбранное устройство в stderr, например `GPU offload: 28 слоёв на CUDA:0 NVIDIA GeForce GTX 1050 Ti (sm_61)`.

Multi-GPU (`-dev 0,1`) - **частично**: слои делятся на непрерывные диапазоны (пропорции задаёт `-tensor-split`), hidden state переезжает через host один раз на границе устройств, веса и своя часть KV-cache кешируются на каждом GPU отдельно. Слои считаются последовательно - параллельного выполнения между устройствами пока нет.

### VRAM в bench

`tools bench` показывает видеопамять через `cuMemGetInfo`: `VRAM: 812 / 4096 MB` в человекочитаемом
выводе и `vram_used_mb` / `vram_total_mb` при `-json`. Для multi-GPU значения суммируются.

### Metal (macOS) - только каркас

`pkg/gpu/metal` - заглушка: `Backend` реализует весь интерфейс `gpu.Backend`, но любой
compute-вызов возвращает `ErrUnavailable` - Metal-kernels ещё нет. Собирается только с `-tags metal`,
`gpu.OpenMetal()` вне macOS отказывает. CUDA-путь это не затрагивает.

## Через Docker

Multi-stage `Dockerfile` запускает `gogguf serve` в минимальном Alpine-образе.

Скачать модель:

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

**CPU (по умолчанию)**:

```bash
docker build -t gogguf .

docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf serve -m /models/Qwen3-0.6B-Q8_0.gguf --host 0.0.0.0:8000
```

**CUDA** (NVIDIA GPU, только `linux/amd64`):

```bash
docker build --target runtime-cuda -t gogguf-cuda .

docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-0.6B-Q8_0.gguf --host 0.0.0.0:8000 -ngl 28
```
