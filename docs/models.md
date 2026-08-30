# Supported models

[Russian version](models-ru.md)

## Works

| Architecture | Status   | Models / notes                                           |
|--------------|----------|----------------------------------------------------------|
| Qwen3        | verified | Qwen3-0.6B, Qwen3-8B, Qwen3-14B                          |
| Qwen2        | code     | `qwen2` via mistral path (Distill-Qwen и т.п.)           |
| Llama 3      | verified | Llama-3.2-1B (CUDA RoPE NORM)                            |
| Llama 2      | code     | same `llama` package; `[INST]`/<<SYS>>, SPM vocab        |
| Mistral      | verified | Mistral-7B-Instruct-v0.2 (NeoX RoPE, GQA; golden Q4_K_M) |
| Mixtral      | code     | `llama` + MoE (8x7B); NeoX via mistral path              |
| Phi-3        | code     | `phi3` / Phi-3.5: fused QKV+FFN, partial RoPE, LongRoPE  |
| Phi-2        | code     | `phi2`: LayerNorm, parallel residual, GELU               |
| Gemma/Gemma2 | code     | `gemma`/`gemma2`: GeGLU; gemma2 softcap/SWA              |
| DeepSeek     | code     | `deepseek`: dense lead + MoE; не `deepseek2` (MLA)       |

Weight formats: **Q8_0**, **Q4_0**, **Q4_1**, **Q5_0**, **Q5_1**, **Q2_K**, **Q3_K**, **Q4_K**, **Q5_K**, **Q6_K**, **Q8_K**.

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

curl -L -o models/Qwen3-8B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf

curl -L -o models/Qwen3-14B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q8_0.gguf

curl -L -o models/Llama-3.2-1B-Instruct-Q8_0.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

curl -L -o models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

```bash
./build/gogguf run -m models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -n 64

./build/gogguf run -m models/Llama-3.2-1B-Instruct-Q8_0.gguf --chat -p "Hello" -n 32

./build/gogguf run -m models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf --chat -p "Hello" -n 32
```

## Soon

1. Vision / multimodal
2. Dedicated embedding models
3. Llama 2 golden fixture
