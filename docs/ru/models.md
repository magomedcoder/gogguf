# Поддерживаемые модели

GoGGUF читает любой GGUF v2/v3 (`gogguf info`, `gogguf inspect`, mmap), но **полный inference** (forward +
генерация) есть только для архитектур по полю `general.architecture` в метаданных GGUF.

## Поддерживается сейчас (end-to-end inference)

| Архитектура        | `general.architecture` | Статус         | Примечания                                                       |
|--------------------|------------------------|----------------|------------------------------------------------------------------|
| **Qwen3**          | `qwen3`                | поддерживается | Основной эталон; full residency; `-b` batch prefill              |
| **Qwen3-MoE**      | `qwen3moe`             | частично       | QK-norm + MoE; 30B-A3B, 235B-A22B; `n_batch` принудительно 1     |
| **Qwen2**          | `qwen2`                | частично       | Distill-Qwen и др.; NeoX через путь mistral                      |
| **Qwen2-MoE**      | `qwen2moe`             | частично       | MoE + gated shared expert (A2.7B, 57B-A14B)                      |
| **Llama 3**        | `llama`                | поддерживается | Forward + chat; golden Llama-3.2-1B; CUDA RoPE NORM              |
| **Mistral**        | `mistral` / `llama`*   | поддерживается | NeoX RoPE, GQA; golden Mistral-7B-v0.2; автоопределение по имени |
| **Mixtral**        | `llama`**              | частично       | MoE 8x7B; `llama.expert_count`; NeoX через mistral               |
| **DeepSeek**       | `deepseek`             | частично       | Dense lead + MoE; не `deepseek2` (MLA)                           |
| **DeepSeek2**      | `deepseek2`            | частично       | Absorbed MLA + YaRN + MoE; MLA на GPU (RoPE/YaRN на CPU)         |
| **Phi-2**          | `phi2`                 | частично       | LayerNorm, parallel residual, GELU                               |
| **Phi-3**          | `phi3`                 | частично       | Fused QKV/FFN, partial RoPE, LongRoPE                            |
| **Gemma / Gemma2** | `gemma` / `gemma2`     | частично       | GeGLU; gemma2 softcap/SWA; у Gemma1 fused residency              |

### Arch * GPU (CUDA `-tags cuda`, `-ngl`)

| Arch                         | Dense residency       | `n_batch`        | Примечания                      |
|------------------------------|-----------------------|------------------|---------------------------------|
| Qwen3 dense                  | full                  | да (default 32)  | fused QKV/AttnFFN/logits        |
| Llama / Mistral              | QKV/AttnFFN           | да               | единый `CanEnableResidency`     |
| Gemma 1                      | fused GeGLU           | да               | без softcap/SWA                 |
| Gemma2                       | частично              | да               | post-norm/softcap/SWA позже     |
| Phi-3                        | FFN + logits          | да               | LongRoPE/fused QKV позже        |
| Phi-2                        | lm_head после host LN | да               | kernels LayerNorm/GELU позже    |
| MoE (Qwen3/Mixtral/DeepSeek) | ExpertFFN + logits    | принудительно 1  | batched experts позже           |
| DeepSeek2                    | MLA KV на device      | принудительно 1  | RoPE/YaRN на CPU                |

\* TheBloke и convert.py часто ставят `general.architecture: llama` - gogguf определяет Mistral по `general.name` / sliding window.

\** Mixtral MoE GGUF: `general.architecture: llama` + `llama.expert_count` > 0.

### Проверенные модели

| Модель                     | Квантизация | Статус                                                         | Ссылки                                                                              |
|----------------------------|-------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Qwen3-1.7B                 | Q8_0        | проверено (golden, layers, logits, CLI, serve, CUDA `-ngl`)    | [Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)                         |
| Llama-3.2-1B-Instruct      | Q8_0        | проверено (golden, layer logits, CLI; CUDA `-ngl` с RoPE NORM) | [Hugging Face](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)         |
| Mistral-7B-Instruct-v0.2   | Q4_K_M      | проверено (golden fixture)                                     | [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)       |

Другие размеры Qwen3 (0.6B, 4B, 8B, ...) с `general.architecture: qwen3` **должны работать**, если веса в поддерживаемой
квантизации (см. ниже). Golden-фикстуры в CI - **1.7B-Q8_0**; в примерах документации часто **4B**.

### Возможности Qwen3

- Prefill + autoregressive decode, KV-cache
- **`n_batch` (`-b`)**: multi-token prefill для **dense** arch (default CLI **32**); MoE/MLA/SWA принудительно `1` (см. [cli.md](cli.md)). На GPU - quant `MatMulMat*` GEMM.
- Полная device residency при `-ngl` = все слои и нативных quant - запуск GPU: [platforms.md](platforms.md), детали: [build.md](build.md)
- Tokenizer BPE (`tokenizer.ggml.model` / qwen2-стиль)
- Chat template: Jinja из GGUF + fallback ChatML/Qwen
- Thinking-блок (`--thinking`)
- Sampling: greedy, temperature, top-k, top-p, min-p, repeat penalty
- HTTP API (`gogguf serve`), REPL (`-i`)

### Поддерживаемые квантизации весов

| Тип                                | CPU     | Native GPU + residency |
|------------------------------------|---------|------------------------|
| F16, F32                           | dequant | путь FP32              |
| Q8_0, Q4_0, Q4_1, Q5_0, Q5_1       | да      | fused                  |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | да      | fused                  |

Также: Legacy / K-quants / без квантизации - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K...Q8_K, F16, F32.

### Планируемые квантизации весов

- Q8_1
- IQ-quants: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- BF16, MXFP4

## GGUF только чтение (без inference)

Для **любой** GGUF-модели:

- `gogguf info` - метаданные, размер, длина контекста
- `gogguf inspect` - список тензоров и типы
- `Load` / `LoadMapped` - разбор и mmap

Если `general.architecture` не поддерживается, inference `Load` возвращает ошибку вида `architecture "..." is not supported`.

## Планируется

| Область                                            | Статус      |
|----------------------------------------------------|-------------|
| Vision (multimodal)                                | планируется |
| Отдельные embedding-модели                         | планируется |
| Metal kernels                                      | планируется |
| Batched MoE / полный residency всех arch           | планируется |

«Embeddings» - отдельные embedding-модели. HTTP `/v1/embeddings` уже отдаёт last-token hidden у генеративных GGUF (см. [api.md](api.md)).

### Сквозные задачи (не привязаны к одной модели)

ОС / GPU и запуск: **[platforms.md](platforms.md)**.

| Задача                                         | Статус                                      |
|------------------------------------------------|---------------------------------------------|
| CUDA Graphs (matmul / layer)                   | сделано                                     |
| Реестр архитектур (`pkg/model/<arch>/`)        | частично (`qwen3`, `llama`, `mistral`, ...) |
| Multi-GPU (`-dev` / P2P, слои последовательно) | частично                                    |
| `forwardBatch` на dense arch                   | сделано                                     |
| Quant GEMM (`MatMulMat*`, fused quants)        | сделано; F32 batch ещё loop-B               |

## Добавление новой архитектуры

1. Реализация в `pkg/model/<arch>/` (config, forward, KV-cache)
2. Регистрация в `pkg/model/model.go` (`switch` по `general.architecture`)
3. Golden-тесты против llama.cpp (см. [testing.md](testing.md))

## Скачать тестовые модели

См. [testing.md](testing.md) - golden/integration-модель и запуск тестов.
