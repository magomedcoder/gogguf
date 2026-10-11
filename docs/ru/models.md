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
| **DeepSeek2**      | `deepseek2`            | частично       | Absorbed MLA + YaRN + MoE (V2/V3 Lite+); MLA на CPU              |
| **Phi-2**          | `phi2`                 | частично       | LayerNorm, parallel residual, GELU                               |
| **Phi-3**          | `phi3`                 | частично       | Fused QKV/FFN, partial RoPE, LongRoPE                            |
| **Gemma / Gemma2** | `gemma` / `gemma2`     | частично       | GeGLU; gemma2 softcap/SWA; без full residency                    |

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
- **`n_batch` (`-b`)**: multi-token prefill **только для dense** Qwen3; MoE принудительно `1` (см. [cli.md](cli.md))
- Полная device residency при `-ngl` = все слои и нативных quant - запуск GPU: [platforms.md](platforms.md), детали: [build.md](build.md)
- Tokenizer BPE (`tokenizer.ggml.model` / qwen2-стиль)
- Chat template: Jinja из GGUF + fallback ChatML/Qwen
- Thinking-блок (`--thinking`)
- Sampling: greedy, temperature, top-k, top-p, min-p, repeat penalty
- HTTP API (`gogguf serve`), REPL (`-i`)

### Поддерживаемые квантизации весов

| Тип                                | CPU     | Native GPU + residency             |
|------------------------------------|---------|------------------------------------|
| F16, F32                           | dequant | путь FP32                          |
| Q8_0, Q4_0, Q4_K, Q5_K, Q6_K       | да      | fused                              |
| Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q8_K | да      | через FP32 dequant (без residency) |

Также: Legacy / K-quants / без квантизации - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K…Q8_K, F16, F32.

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
| True batched GPU GEMM; `forwardBatch` на все dense | планируется |
| Metal kernels; CUDA на Windows                     | планируется |

«Embeddings» - отдельные embedding-модели. HTTP `/v1/embeddings` уже отдаёт last-token hidden у генеративных GGUF (см. [api.md](api.md)).

### Сквозные задачи (не привязаны к одной модели)

ОС / GPU и запуск: **[platforms.md](platforms.md)**.

| Задача                                  | Статус                                      |
|-----------------------------------------|---------------------------------------------|
| CUDA Graphs (matmul / layer)            | сделано                                     |
| Реестр архитектур (`pkg/model/<arch>/`) | частично (`qwen3`, `llama`, `mistral`, ...) |
| Multi-GPU pipeline split                | частично (`-dev` / `-tensor-split`)         |
| `forwardBatch` на не-Qwen3              | не сделано                                  |
| True batched GPU GEMM                   | не сделано                                  |

## Добавление новой архитектуры

1. Реализация в `pkg/model/<arch>/` (config, forward, KV-cache)
2. Регистрация в `pkg/model/model.go` (`switch` по `general.architecture`)
3. Golden-тесты против llama.cpp (см. [testing.md](testing.md))

## Скачать тестовые модели

См. [testing.md](testing.md) - golden/integration-модель и запуск тестов.
