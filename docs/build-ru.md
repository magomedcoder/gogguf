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

На GPU: matmul (Q8_0 и FP32), FFN residency (gate/up/SwiGLU/down), attention (+ softmax), KV-cache. RMSNorm/RoPE на CPU (PCIe).

Blackwell (sm_120, RTX 50xx): нужен PTX 8.7+; Q8_0 scale конвертируется в FP32 при загрузке на GPU (без PTX f16).

Без `-tags cuda` при `-ngl > 0` будет ошибка `gpu: CUDA недоступна`.

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

docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf serve -m /models/Qwen3-0.6B-Q8_0.gguf --addr 0.0.0.0:8000
```

**CUDA** (NVIDIA GPU, только `linux/amd64`):

```bash
docker build --target runtime-cuda -t gogguf-cuda .

docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-0.6B-Q8_0.gguf --addr 0.0.0.0:8000 -ngl 28
```
