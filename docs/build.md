# Build

## Local build

Requires **Go 1.27+**.

```bash
go build -o build/gogguf ./cmd/gogguf
```

**GPU / OS support and how to run with `-ngl`:** see **[platforms.md](platforms.md)**.

## CUDA build (optional)

On **Linux + NVIDIA** only (details: [platforms.md](platforms.md)):

```bash
CGO_ENABLED=1 go build -tags cuda -o build/gogguf ./cmd/gogguf
CGO_ENABLED=1 go test -tags=cuda ./pkg/device/cuda/...
```

### What actually runs on GPU

For **Qwen3 dense** with `-ngl` = all layers (Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K weights) the residual stream is device-resident:

| Stage                                                      | Where                    | Note                                           |
|------------------------------------------------------------|--------------------------|------------------------------------------------|
| tokenizer, chat template                                   | CPU                      | -                                              |
| token embedding                                            | CPU + one HtoD per token | `HiddenUpload` writes `d_resid`                |
| `attn_norm` + QKV + QK-norm + RoPE + KV append + attention | GPU                      | fused `QKVRoPEAttention*Cached`, CUDA Graph    |
| WO + residual + `ffn_norm` + FFN (SwiGLU) + residual       | GPU                      | fused `AttnFFNResidual*Cached`, CUDA Graph     |
| residual `x` between layers                                | GPU (`d_resid`)          | no PCIe traffic per layer                      |
| `output_norm` + `lm_head`                                  | GPU                      | `LogitsFromDevice`, one DtoH of `vocab` floats |
| sampling, KV mirror, HTTP                                  | CPU                      | `attn`/`k`/`v` of a layer still go DtoH        |

Partly on GPU: F32 batch still device-offset vec launches; MoE / DeepSeek2 MLA accept `-b>1` via serial per-token resid inside chunks; Mistral SWA `-b>1` with windowed causal attn (reclaim still sets `gpuKVStale` until A4). Quant prefill uses `MatMulMat*` (Q8_0 single-launch PTX).  
**`gpuKVStale`:** if GPU KV append fails or SWA reclaim drops host KV, fused GPU attention is skipped until `ResetCache()` / a new session.  
Fused native: Q8_0, Q8_1, Q4_0/1, Q5_0/1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K (`FusedQuantSupported`). IQ-*/MXFP4 are dtype-registered for Load with CPU dequant stubs (no residency). BF16 dequants on CPU via Floats path.

Flags (`-ngl`, `-dev`, Docker GPU, macOS/Windows): **[platforms.md](platforms.md)**.

## Docker

Multi-stage `Dockerfile` runs `gogguf serve` in a minimal Alpine image.

```bash
mkdir -p models
curl -L -o models/Qwen3-4B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf

docker build -t gogguf .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" gogguf serve -m /models/Qwen3-4B-Q8_0.gguf --host 0.0.0.0:8000
```

**GPU Docker** (NVIDIA, `linux/amd64`): see [platforms.md](platforms.md#docker-gpu).
