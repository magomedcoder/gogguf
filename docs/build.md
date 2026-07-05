# Build

[Русская версия](build-ru.md)

## Local build

```bash
go build -o build/gogguf ./cmd/gogguf
```

No CGO required - cross-compile to any platform.

## CUDA (NVIDIA GPU, optional)

Requires an NVIDIA driver (`libcuda.so`) and CGO.

CUDA Toolkit is not required - Driver API is used via `dlopen`.

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf
```

Verify GPU matmul:

```bash
CGO_ENABLED=1 go test -tags=cuda ./pkg/gpu/cuda/...
```

`-ngl N` - matmul for the first N transformer layers on GPU (max `block_count` from `gguf inspect`; Qwen3-0.6B - 28).

Currently only matmul runs on GPU: Q8_0 without FP32 dequantization, other types via FP32. Attention, norm, and RoPE stay on CPU. Q8_0 kernel requires GPU sm_70+.

Without `-tags cuda`, `-ngl > 0` returns `gpu: CUDA unavailable`.

## Docker

Multi-stage `Dockerfile`: CPU cross-compilation by default, separate CUDA targets.

**CPU (default)** - all platforms, `CGO_ENABLED=0`:

```bash
docker build -t gogguf-build .
docker run --rm -v "$(pwd)/build:/out" gogguf-build
```

**CUDA** - `linux-amd64/gguf-cuda` only:

```bash
docker build --target cuda -t gogguf-cuda .
docker run --rm -v "$(pwd)/build:/out" gogguf-cuda
```

**CPU + CUDA**:

```bash
docker build --target release -t gogguf-release .
docker run --rm -v "$(pwd)/build:/out" gogguf-release
```

| Target / mode | Output                         |
|---------------|--------------------------------|
| *(default)*   | CPU binaries for all platforms |
| `cuda`        | `build/linux-amd64/gguf-cuda`  |
| `release`     | CPU + `gguf-cuda`              |

| Platform      | CPU binary                     | CUDA binary (`cuda` / `release`) |
|---------------|--------------------------------|----------------------------------|
| Linux amd64   | `build/linux-amd64/gguf`       | `build/linux-amd64/gguf-cuda`    |
| Linux arm64   | `build/linux-arm64/gguf`       | -                                |
| Windows amd64 | `build/windows-amd64/gguf.exe` | -                                |
| Windows arm64 | `build/windows-arm64/gguf.exe` | -                                |
| macOS amd64   | `build/darwin-amd64/gguf`      | -                                |
| macOS arm64   | `build/darwin-arm64/gguf`      | -                                |

> **Note.** Binary path depends on the build method:
> - local CPU: `./build/gguf`
> - local CUDA: `./build/gguf` (with `-tags cuda`)
> - Docker CPU: `./build/<os>-<arch>/gguf` (`.exe` on Windows)
> - Docker CUDA: `./build/linux-amd64/gguf-cuda`
