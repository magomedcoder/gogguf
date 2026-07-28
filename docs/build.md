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

`-ngl N` - offload the first N transformer layers to GPU (max `block_count`; Qwen3-0.6B - 28).

On GPU: matmul (FP32, Q8_0, Q4_0) with CUDA Graphs (full HtoD+kernel+DtoH, or kernel-only when the same vec is already resident), FFN residency, QKV+RoPE+attn residency, attn+FFN residual (WO+RMSNorm+FFN, parallel RMSNorm apply), attention (+ softmax), KV-cache. Disable with `GGUF_QKV_RESIDENCY=0` / `GGUF_ATTN_FFN_RESIDENCY=0`. Call `Engine.Close()` to free VRAM between loads.

Blackwell (sm_120, RTX 50xx): requires PTX 8.7+; Q8_0 scales are converted to FP32 on upload (no PTX f16).

Without `-tags cuda`, `-ngl > 0` returns `gpu: CUDA unavailable`.

## Docker

Multi-stage `Dockerfile` runs `gogguf serve` in a minimal Alpine image.

Download a model:

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

**CPU (default)**:

```bash
docker build -t gogguf .

docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf serve -m /models/Qwen3-0.6B-Q8_0.gguf --addr 0.0.0.0:8000
```

**CUDA** (NVIDIA GPU, `linux/amd64` only):

```bash
docker build --target runtime-cuda -t gogguf-cuda .

docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-0.6B-Q8_0.gguf --addr 0.0.0.0:8000 -ngl 28
```
