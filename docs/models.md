# Supported models

[Russian version](models-ru.md)

GoGGUF can read any GGUF v2/v3 file (`gogguf info`, `gogguf inspect`, mmap), but **full inference** (forward pass +
generation) is implemented only for specific architectures - selected by `general.architecture` in GGUF metadata.

## Supported today (end-to-end inference)

| Architecture  | `general.architecture` | Status    | Notes                                                       |
|---------------|------------------------|-----------|-------------------------------------------------------------|
| **Qwen3**     | `qwen3`                | supported | Primary reference                                           |
| **Qwen3-MoE** | `qwen3moe`             | partial   | QK-norm + MoE; 30B-A3B, 235B-A22B                           |
| **Qwen2**     | `qwen2`                | partial   | Distill-Qwen etc.; NeoX via mistral path                    |
| **Qwen2-MoE** | `qwen2moe`             | partial   | MoE + gated shared expert (A2.7B, 57B-A14B)                 |
| **Llama 3**   | `llama`                | supported | Forward + chat; golden Llama-3.2-1B; CUDA RoPE NORM         |
| **Mistral**   | `mistral` / `llama`*   | supported | NeoX RoPE, GQA; golden Mistral-7B-v0.2; auto-detect by name |
| **Mixtral**   | `llama`**              | partial   | MoE 8x7B; `llama.expert_count`; NeoX via mistral            |
| **DeepSeek**  | `deepseek`             | partial   | Dense lead + MoE; not `deepseek2` (MLA)                     |
| **DeepSeek2** | `deepseek2`            | partial   | Absorbed MLA + YaRN + MoE (V2/V3 Lite+)                     |

\* TheBloke and convert.py often set `general.architecture: llama` - gogguf detects Mistral via `general.name` / sliding
window.

\** Mixtral MoE GGUF: `general.architecture: llama` + `llama.expert_count` > 0.

### Llama 3 - verified

| Model                 | Quantization | Status                                                           | Links                                                                       |
|-----------------------|--------------|------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Llama-3.2-1B-Instruct | Q8_0         | verified (golden, layer logits, CLI; CUDA `-ngl` with RoPE NORM) | [Hugging Face](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |

### Qwen3 - verified

| Model      | Quantization | Status                                           | Links                                                       |
|------------|--------------|--------------------------------------------------|-------------------------------------------------------------|
| Qwen3-0.6B | Q8_0         | verified (golden tests, CLI, serve, CUDA `-ngl`) | [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |

Other Qwen3 sizes (1.7B, 4B, 8B, ...) with `general.architecture: qwen3` **should work** if weights use a supported
quantization (see below). They are not yet covered by automated golden tests.

### Qwen3 capabilities

- Prefill + autoregressive decode, KV-cache; **n_batch** (`-b`) for CPU multi-token prefill
- BPE tokenizer (`tokenizer.ggml.model` / qwen2-style)
- Chat template: Jinja from GGUF + ChatML/Qwen fallback
- Thinking block (`--thinking`)
- Sampling: greedy, temperature, top-k, top-p, min-p, repeat penalty
- HTTP API (`gogguf serve`), REPL (`-i`)
- CUDA offload (`-ngl`, `-tags cuda` build): matmul, RMSNorm, RoPE, SwiGLU, attention, GPU KV-cache

### Supported weight quantizations

| Type | Status    |
|------|-----------|
| Q8_0 | supported |
| Q4_0 | supported |
| Q4_1 | supported |
| Q5_0 | supported |
| Q5_1 | supported |
| Q2_K | supported |
| Q3_K | supported |
| Q4_K | supported |
| Q5_K | supported |
| Q6_K | supported |
| Q8_K | supported |

## GGUF read-only (no inference)

For **any** GGUF model:

- `gogguf info` - metadata, size, context length
- `gogguf inspect` - tensor list and types
- `Load` / `LoadMapped` - parsing and mmap

If `general.architecture` is not supported, inference `Load` returns an error like `architecture "..." is not supported`.

## Planned

Priority per [roadmap.md](roadmap.md):

| Priority | Architecture / area | Expected `general.architecture` | Status    |
|----------|---------------------|---------------------------------|-----------|
| 1        | Mistral             | `mistral`                       | supported |
| 2        | Llama 2             | `llama`                         | partial   |
| 3        | Phi-3               | `phi3`                          | partial   |
| 3b       | Phi-2               | `phi2`                          | partial   |
| 4        | Gemma / Gemma2      | `gemma` / `gemma2`              | partial   |
| -        | Vision (multimodal) | -                               | planned   |
| -        | Embeddings          | -                               | planned   |

### Cross-cutting (not tied to one model)

| Task                                        | Status                                        |
|---------------------------------------------|-----------------------------------------------|
| CUDA Graphs (matmul)                        | done                                          |
| CUDA Graphs (layer)                         | done (FFN, residual, QKV+RoPE, attn, head)    |
| Metal (macOS)                               | scaffold only (`pkg/gpu/metal`, no kernels)   |
| Multi-GPU (`-dev N`)                        | `-dev N` done; layer split `-dev 0,1` partial |
| More quantizations (Q5_K, Q8_K, ...)        | done                                          |
| Architecture registry (`pkg/model/<arch>/`) | partial (`qwen3`, `llama`, `mistral`, ...)    |

## Adding a new architecture

1. Implement `pkg/model/<arch>/` (config, forward, KV-cache)
2. Register in `pkg/model/model.go` (`switch` on `general.architecture`)
3. Golden tests vs llama.cpp (see [testing.md](testing.md))

## Download test models

```bash
mkdir -p models

# Qwen3 (primary golden model)
curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Llama 3.2 1B Instruct
curl -L -o models/Llama-3.2-1B-Instruct-Q8_0.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Mistral 7B Instruct v0.2
curl -L -o models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```
