# Сборка

## Локально

Нужен **Go 1.27+**.

```bash
go build -o build/gogguf ./cmd/gogguf
```

**Поддержка ОС / GPU и запуск с `-ngl`:** см. **[platforms.md](platforms.md)**.

## Сборка CUDA (опционально)

**Linux + NVIDIA** и **Windows** (`nvcuda.dll`; подробности: [platforms.md](platforms.md)):

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf
CGO_ENABLED=1 go test -tags=cuda ./pkg/device/cuda/...
```

### Что реально считается на GPU

Для **Qwen3 dense** при `-ngl` = все слои и fused-квантах (Q8_0 / Q4_0/1 / Q5_0/1 / Q2_K...Q8_K) residual stream живёт на устройстве:

| Этап                                                       | Где                      | Примечание                                     |
|------------------------------------------------------------|--------------------------|------------------------------------------------|
| tokenizer, chat template                                   | CPU                      | -                                              |
| embedding токена                                           | CPU + один HtoD на токен | `HiddenUpload` пишет в `d_resid`               |
| `attn_norm` + QKV + QK-norm + RoPE + KV append + attention | GPU                      | fused `QKVRoPEAttention*Cached`, CUDA Graph    |
| WO + residual + `ffn_norm` + FFN (SwiGLU) + residual       | GPU                      | fused `AttnFFNResidual*Cached`, CUDA Graph     |
| residual `x` между слоями                                  | GPU (`d_resid`)          | PCIe на слой нет                               |
| `output_norm` + `lm_head`                                  | GPU                      | `LogitsFromDevice`, один DtoH на `vocab` float |
| sampling, зеркало KV, HTTP                                 | CPU                      | `attn`/`k`/`v` слоя пока уходят DtoH           |

Частично на GPU: F32 batch - device-offset vec launches; MoE / DeepSeek2 MLA / SWA -> `n_batch=1`; не у всех arch полный residency. Quant prefill - `MatMulMat*` (Q8_0 single-launch PTX).  
**`gpuKVStale`:** при сбое GPU KV fused attention отключается до `ResetCache()` / новой сессии.  
Fused native: Q8_0, Q4_0/1, Q5_0/1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K (`FusedQuantSupported`). Остальные (Q8_1, IQ-*) -> FP32 и residency выкл.

Флаги (`-ngl`, `-dev`, Docker GPU, macOS/Windows): **[platforms.md](platforms.md)**.

## Docker

Multi-stage `Dockerfile` запускает `gogguf serve` в минимальном Alpine-образе.

```bash
mkdir -p models
curl -L -o models/Qwen3-4B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf

docker build -t gogguf .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf serve -m /models/Qwen3-4B-Q8_0.gguf --host 0.0.0.0:8000
```

**Docker с GPU** (NVIDIA, `linux/amd64`): см. [platforms.md](platforms.md#docker-gpu).
