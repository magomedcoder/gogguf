# Supported models

GoGGUF can read any GGUF v2/v3 file (`gogguf info`, `gogguf inspect`, mmap), but **full inference** (forward pass +
generation) is implemented only for specific architectures - selected by `general.architecture` in GGUF metadata.

## Supported today (end-to-end inference)

| Architecture       | `general.architecture` | Status    | Notes                                                       |
|--------------------|------------------------|-----------|-------------------------------------------------------------|
| **Qwen3**          | `qwen3`                | supported | Primary reference; full residency; `-b` batch prefill       |
| **Qwen3-MoE**      | `qwen3moe`             | partial   | QK-norm + MoE; 30B-A3B, 235B-A22B; `n_batch` forced to 1    |
| **Qwen2**          | `qwen2`                | partial   | Distill-Qwen etc.; NeoX via mistral path                    |
| **Qwen2-MoE**      | `qwen2moe`             | partial   | MoE + gated shared expert (A2.7B, 57B-A14B)                 |
| **Llama 3**        | `llama`                | supported | Forward + chat; golden Llama-3.2-1B; CUDA RoPE NORM         |
| **Mistral**        | `mistral` / `llama`*   | supported | NeoX RoPE, GQA; golden Mistral-7B-v0.2; auto-detect by name |
| **Mixtral**        | `llama`**              | partial   | MoE 8x7B; `llama.expert_count`; NeoX via mistral            |
| **DeepSeek**       | `deepseek`             | partial   | Dense lead + MoE; not `deepseek2` (MLA)                     |
| **DeepSeek2**      | `deepseek2`            | partial   | Absorbed MLA + YaRN + MoE (V2/V3 Lite+); MLA on CPU         |
| **Phi-2**          | `phi2`                 | partial   | LayerNorm, parallel residual, GELU                          |
| **Phi-3**          | `phi3`                 | partial   | Fused QKV/FFN, partial RoPE, LongRoPE                       |
| **Gemma / Gemma2** | `gemma` / `gemma2`     | partial   | GeGLU; gemma2 softcap/SWA; no full residency                |

\* TheBloke and convert.py often set `general.architecture: llama` - gogguf detects Mistral via `general.name` / sliding window.

\** Mixtral MoE GGUF: `general.architecture: llama` + `llama.expert_count` > 0.

### Verified models

| Model                      | Quantization | Status                                                           | Links                                                                               |
|----------------------------|--------------|------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Qwen3-1.7B                 | Q8_0         | verified (golden, layers, logits, CLI, serve, CUDA `-ngl`)       | [Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)                         |
| Llama-3.2-1B-Instruct      | Q8_0         | verified (golden, layer logits, CLI; CUDA `-ngl` with RoPE NORM) | [Hugging Face](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)         |
| Mistral-7B-Instruct-v0.2   | Q4_K_M       | verified (golden fixture)                                        | [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)       |

Other Qwen3 sizes (0.6B, 4B, 8B, ...) with `general.architecture: qwen3` **should work** if weights use a supported
quantization (see below). Golden fixtures in CI cover **1.7B-Q8_0**; docs/examples often use **4B** for illustration.

### Qwen3 capabilities

- Prefill + autoregressive decode, KV-cache
- **`n_batch` (`-b`)**: multi-token prefill for **dense** Qwen3 only; MoE forced to `1` (see [cli.md](cli.md))
- Full device residency when `-ngl` = all layers and native quants - GPU launch: [platforms.md](platforms.md), internals: [build.md](build.md)
- BPE tokenizer (`tokenizer.ggml.model` / qwen2-style)
- Chat template: Jinja from GGUF + ChatML/Qwen fallback
- Thinking block (`--thinking`)
- Sampling: greedy, temperature, top-k, top-p, min-p, repeat penalty
- HTTP API (`gogguf serve`), REPL (`-i`)

### Supported weight quantizations

| Type                               | CPU     | Native GPU + residency          |
|------------------------------------|---------|---------------------------------|
| F16, F32                           | dequant | FP32 path                       |
| Q8_0, Q4_0, Q4_K, Q5_K, Q6_K       | yes     | fused                           |
| Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q8_K | yes     | via FP32 dequant (no residency) |

Also listed as Legacy / K-quants / unquantized in tooling: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K…Q8_K, F16, F32.

### Planned weight quantizations

- Q8_1
- IQ-quants: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- BF16, MXFP4

## GGUF read-only (no inference)

For **any** GGUF model:

- `gogguf info` - metadata, size, context length
- `gogguf inspect` - tensor list and types
- `Load` / `LoadMapped` - parsing and mmap

If `general.architecture` is not supported, inference `Load` returns an error like `architecture "..." is not supported`.

## Planned

| Area                                               | Status  |
|----------------------------------------------------|---------|
| Vision (multimodal)                                | planned |
| Dedicated embedding models                         | planned |
| True batched GPU GEMM; `forwardBatch` on all dense | planned |
| Metal kernels; Windows CUDA                        | planned |

“Embeddings” means dedicated embedding models. HTTP `/v1/embeddings` already returns last-token hidden for generative GGUF (see [api.md](api.md)).

### Cross-cutting (not tied to one model)

OS / GPU support and launch: **[platforms.md](platforms.md)**.

| Task                                        | Status                                     |
|---------------------------------------------|--------------------------------------------|
| CUDA Graphs (matmul / layer)                | done                                       |
| Architecture registry (`pkg/model/<arch>/`) | partial (`qwen3`, `llama`, `mistral`, ...) |
| Multi-GPU pipeline split                    | partial (`-dev` / `-tensor-split`)         |
| `forwardBatch` on non-Qwen3                 | not done                                   |
| True batched GPU GEMM                       | not done                                   |

## Adding a new architecture

1. Implement `pkg/model/<arch>/` (config, forward, KV-cache)
2. Register in `pkg/model/model.go` (`switch` on `general.architecture`)
3. Golden tests vs llama.cpp (see [testing.md](testing.md))

## Download test models

See [testing.md](testing.md) for the golden/integration model and how to run tests.
