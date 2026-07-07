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

На GPU: matmul (Q8_0 и FP32), RMSNorm, RoPE, attention, SwiGLU.

Blackwell (sm_120, RTX 50xx): нужен PTX 8.7+; Q8_0 scale конвертируется в FP32 при загрузке на GPU (без PTX f16).

Без `-tags cuda` при `-ngl > 0` будет ошибка `gpu: CUDA недоступна`.

## Через Docker

Multi-stage `Dockerfile`: CPU-кросс-компиляция по умолчанию, отдельные target'ы для CUDA.

**CPU (по умолчанию)** - все платформы, `CGO_ENABLED=0`:

```bash
docker build -t gogguf-build .
docker run --rm -v "$(pwd)/build:/out" gogguf-build
```

**CUDA** - только `linux-amd64/gguf-cuda`:

```bash
docker build --target cuda -t gogguf-cuda .
docker run --rm -v "$(pwd)/build:/out" gogguf-cuda
```

**CPU + CUDA**:

```bash
docker build --target release -t gogguf-release .
docker run --rm -v "$(pwd)/build:/out" gogguf-release
```

| Target / режим | Результат                     |
|----------------|-------------------------------|
| *(default)*    | CPU-бинарники всех платформ   |
| `cuda`         | `build/linux-amd64/gguf-cuda` |
| `release`      | CPU + `gguf-cuda`             |

| Платформа     | CPU-бинарник                   | CUDA-бинарник (target `cuda` / `release`) |
|---------------|--------------------------------|-------------------------------------------|
| Linux amd64   | `build/linux-amd64/gguf`       | `build/linux-amd64/gguf-cuda`             |
| Linux arm64   | `build/linux-arm64/gguf`       | -                                         |
| Windows amd64 | `build/windows-amd64/gguf.exe` | -                                         |
| Windows arm64 | `build/windows-arm64/gguf.exe` | -                                         |
| macOS amd64   | `build/darwin-amd64/gguf`      | -                                         |
| macOS arm64   | `build/darwin-arm64/gguf`      | -                                         |

> **Примечание.** Путь к бинарнику зависит от способа сборки:
> - локально CPU: `./build/gguf`
> - локально CUDA: `./build/gguf` (с `-tags cuda`)
> - Docker CPU: `./build/<os>-<arch>/gguf` (на Windows: `gguf.exe`)
> - Docker CUDA: `./build/linux-amd64/gguf-cuda`
