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

On GPU: matmul (FP32, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K) with CUDA Graphs (full HtoD+kernel+DtoH, or kernel-only when the same vec is already resident), layer graphs for FFN, attn/FFN residual, QKV+RoPE prefix, and attention (keyed by `seq_len`; KV append stays discrete), FFN residency, QKV+RoPE+attn residency, attn+FFN residual (WO+RMSNorm+FFN, parallel RMSNorm apply), attention (+ softmax), KV-cache. Disable with `GGUF_QKV_RESIDENCY=0` / `GGUF_ATTN_FFN_RESIDENCY=0`. Call `Engine.Close()` to free VRAM between loads.

Blackwell (sm_120, RTX 50xx): requires PTX 8.7+; Q8_0 scales are converted to FP32 on upload (no PTX f16).

Without `-tags cuda`, `-ngl > 0` returns `gpu: CUDA unavailable`.

### What actually runs on GPU

For **Qwen3 dense** with `-ngl` = all layers (Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K weights) the whole residual stream is device-resident:

| Stage                                                      | Where                    | Note                                           |
|------------------------------------------------------------|--------------------------|------------------------------------------------|
| tokenizer, chat template                                   | CPU                      | -                                              |
| token embedding                                            | CPU + one HtoD per token | `HiddenUpload` writes `d_resid`                |
| `attn_norm` + QKV + QK-norm + RoPE + KV append + attention | GPU                      | fused `QKVRoPEAttention*Cached`, CUDA Graph    |
| WO + residual + `ffn_norm` + FFN (SwiGLU) + residual       | GPU                      | fused `AttnFFNResidual*Cached`, CUDA Graph     |
| residual `x` between layers                                | GPU (`d_resid`)          | no PCIe traffic per layer                      |
| `output_norm` + `lm_head`                                  | GPU                      | `LogitsFromDevice`, one DtoH of `vocab` floats |
| sampling, KV mirror, HTTP                                  | CPU                      | `attn`/`k`/`v` of a layer still go DtoH        |

Partly on GPU: `-b > 1` prefill (chunk matmul is CPU-GEMM, K/V mirrored into GPU KV), other architectures (Llama / Mistral: matmul + FFN + attention, no full residency), MoE experts (CPU), DeepSeek2 MLA (CPU only).
Weight types without native kernels (Q4_1, Q5_0/1, Q2_K, Q3_K, Q8_K) fall back to the FP32 path and disable residency.

### Device selection and multi-GPU

```bash
./build/gogguf run   -m model.gguf -ngl 28 -dev 1 -p "Hi" # second visible GPU
./build/gogguf serve -m model.gguf -ngl 28 -dev 0,1 # split layers across two GPUs
./build/tools bench  -m model.gguf -ngl 28 -dev 0,1 -tensor-split 0.75,0.25
CUDA_VISIBLE_DEVICES=1 ./build/gogguf run -m model.gguf -ngl 28 -p "Hi"
```

`-dev` takes ordinals **after** `CUDA_VISIBLE_DEVICES` filtering (`-dev 0` is the first visible device), and an out-of-range ordinal is rejected at init. `run` / `serve` / `tools bench` print the  chosen device to stderr, e.g. `GPU offload: 28 layers on CUDA:0 NVIDIA GeForce GTX 1050 Ti (sm_61)`.

Multi-GPU (`-dev 0,1`) is **partial**: layers are split into contiguous ranges (optionally weighted by `-tensor-split`), the hidden state crosses the device boundary through host memory once per token, and each device caches its own weights and its own slice of the KV-cache. Layers run sequentially - there is no parallel execution across devices yet.

### VRAM in bench

`tools bench` reports video memory via `cuMemGetInfo`: `VRAM: 812 / 4096 MB` in human output and
`vram_used_mb` / `vram_total_mb` with `-json`. For multi-GPU the numbers are summed over devices.

### Metal (macOS) - scaffold only

`pkg/gpu/metal` is a placeholder: `Backend` implements the full `gpu.Backend` interface, but every
compute call returns `ErrUnavailable` - there are no Metal kernels yet. It is built only with
`-tags metal` and `gpu.OpenMetal()` fails on non-Darwin platforms. The CUDA path is unaffected.

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

docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf serve -m /models/Qwen3-0.6B-Q8_0.gguf --host 0.0.0.0:8000
```

**CUDA** (NVIDIA GPU, `linux/amd64` only):

```bash
docker build --target runtime-cuda -t gogguf-cuda .

docker run --rm --gpus all -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf-cuda serve -m /models/Qwen3-0.6B-Q8_0.gguf --host 0.0.0.0:8000 -ngl 28
```
