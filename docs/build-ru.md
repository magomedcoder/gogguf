# Сборка

[English version](build.md)

## Локально

```bash
go build -o build/gguf ./cmd/gguf
```

Без CGO - кросс-компиляция на любую платформу.

## CUDA (NVIDIA GPU, опционально)

Требуется драйвер NVIDIA (`libcuda.so`) и CGO.

CUDA Toolkit не нужен - используется Driver API через `dlopen`.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gguf ./cmd/gguf
```

Проверка GPU matmul:

```bash
CGO_ENABLED=1 go test -tags=cuda ./pkg/gpu/cuda/...
```

`-ngl N` - matmul первых N transformer-слоёв на GPU (макс. `block_count` из `gguf inspect`; Qwen3-0.6B - 28).

Сейчас на GPU только matmul: Q8_0 без деквантизации в FP32, остальные типы - через FP32. Attention, norm и RoPE - на
CPU. Q8_0 kernel требует GPU sm_70+.

Без `-tags cuda` при `-ngl > 0` будет ошибка `gpu: CUDA недоступна`.

## Через Docker

Multi-stage `Dockerfile`: CPU-кросс-компиляция по умолчанию, отдельные target'ы для CUDA.

**CPU (по умолчанию)** - все платформы, `CGO_ENABLED=0`:

```bash
docker build -t gguf-build .
docker run --rm -v "$(pwd)/build:/out" gguf-build
```

**CUDA** - только `linux-amd64/gguf-cuda`:

```bash
docker build --target cuda -t gguf-cuda .
docker run --rm -v "$(pwd)/build:/out" gguf-cuda
```

**CPU + CUDA**:

```bash
docker build --target release -t gguf-release .
docker run --rm -v "$(pwd)/build:/out" gguf-release
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
