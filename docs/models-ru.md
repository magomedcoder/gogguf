# Поддерживаемые модели

[English version](models.md)

GoGGUF читает любой GGUF v2/v3 (`gogguf info`, `gogguf inspect`, mmap), но **полный inference** (forward + генерация) есть только для архитектур по полю `general.architecture` в метаданных GGUF.

## Поддерживается сейчас (end-to-end inference)

| Архитектура   | `general.architecture` | Статус         | Примечания                                                       |
|---------------|------------------------|----------------|------------------------------------------------------------------|
| **Qwen3**     | `qwen3`                | поддерживается | Основной эталон                                                  |
| **Qwen3-MoE** | `qwen3moe`             | частично       | QK-norm + MoE; 30B-A3B, 235B-A22B                                |
| **Qwen2**     | `qwen2`                | частично       | Distill-Qwen и др.; NeoX через путь mistral                      |
| **Qwen2-MoE** | `qwen2moe`             | частично       | MoE + gated shared expert (A2.7B, 57B-A14B)                      |
| **Llama 3**   | `llama`                | поддерживается | Forward + chat; golden Llama-3.2-1B; CUDA RoPE NORM              |
| **Mistral**   | `mistral` / `llama`*   | поддерживается | NeoX RoPE, GQA; golden Mistral-7B-v0.2; автоопределение по имени |
| **Mixtral**   | `llama`**              | частично       | MoE 8x7B; `llama.expert_count`; NeoX через mistral               |
| **DeepSeek**  | `deepseek`             | частично       | Dense lead + MoE; не `deepseek2` (MLA)                           |
| **DeepSeek2** | `deepseek2`            | частично       | Absorbed MLA + YaRN + MoE (V2/V3 Lite+)                          |

\* TheBloke и convert.py часто ставят `general.architecture: llama` - gogguf определяет Mistral по `general.name` / sliding window.

\** Mixtral MoE GGUF: `general.architecture: llama` + `llama.expert_count` > 0.

### Llama 3 - проверено

| Модель                | Квантизация | Статус                                                               | Ссылки                                                                      |
|-----------------------|-------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Llama-3.2-1B-Instruct | Q8_0        | проверено (golden, layer logits, CLI; CUDA `-ngl` с RoPE NORM)       | [Hugging Face](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |

### Qwen3 - проверено

| Модель     | Квантизация | Статус                                              | Ссылки                                                      |
|------------|-------------|-----------------------------------------------------|-------------------------------------------------------------|
| Qwen3-0.6B | Q8_0        | проверено (golden-тесты, CLI, serve, CUDA `-ngl`)   | [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |

Другие размеры Qwen3 (1.7B, 4B, 8B, ...) с `general.architecture: qwen3` **должны работать**, если веса в поддерживаемой квантизации (см. ниже). Автоматических golden-тестов для них пока нет.

### Возможности Qwen3

- Prefill + авторегрессионный decode, KV-cache; **n_batch** (`-b`) - multi-token prefill на CPU
- Tokenizer BPE (`tokenizer.ggml.model` / qwen2-стиль)
- Chat template: Jinja из GGUF + fallback ChatML/Qwen
- Thinking-блок (`--thinking`)
- Sampling: greedy, temperature, top-k, top-p, min-p, repeat penalty
- HTTP API (`gogguf serve`), REPL (`-i`)
- CUDA offload (`-ngl`, сборка `-tags cuda`): matmul, RMSNorm, RoPE, SwiGLU, attention, GPU KV-cache

### Поддерживаемые квантизации весов

| Тип                                   | Статус         |
|---------------------------------------|----------------|
| Q8_0                                  | поддерживается |
| Q4_0                                  | поддерживается |
| Q4_K                                  | поддерживается |
| Q6_K                                  | поддерживается |
| Q4_1, Q5_0, Q5_1, Q2_K ... Q5_K, Q8_K | планируется    |

## GGUF только чтение (без inference)

Для **любой** GGUF-модели:

- `gogguf info` - метаданные, размер, длина контекста
- `gogguf inspect` - список тензоров и типы
- `Load` / `LoadMapped` - разбор и mmap

Если `general.architecture` не поддерживается, inference `Load` возвращает ошибку вида `architecture "..." is not supported`.

## Планируется

Приоритет по [roadmap.md](roadmap.md):

| Приоритет | Архитектура / область | Ожидаемый `general.architecture` | Статус         |
|-----------|-----------------------|----------------------------------|----------------|
| 1         | Mistral               | `mistral`                        | поддерживается |
| 2         | Llama 2               | `llama`                          | частично       |
| 3         | Phi-3                 | `phi3`                           | частично       |
| 3b        | Phi-2                 | `phi2`                           | частично       |
| 4         | Gemma / Gemma2        | `gemma` / `gemma2`               | частично       |
| -         | Vision (multimodal)   | -                                | планируется    |
| -         | Embeddings            | -                                | планируется    |

### Сквозные задачи (не привязаны к одной модели)

| Задача                                  | Статус                                      |
|-----------------------------------------|---------------------------------------------|
| CUDA Graphs (matmul)                    | сделано                                     |
| CUDA Graphs (layer)                     | планируется                                 |
| Metal (macOS)                           | планируется                                 |
| Multi-GPU (`-dev N`)                    | планируется                                 |
| Больше квантизаций (Q5_K, Q8_K, ...)    | планируется                                 |
| Реестр архитектур (`pkg/model/<arch>/`) | частично (`qwen3`, `llama`, `mistral`, ...) |

## Добавление новой архитектуры

1. Реализация в `pkg/model/<arch>/` (config, forward, KV-cache)
2. Регистрация в `pkg/model/model.go` (`switch` по `general.architecture`)
3. Golden-тесты против llama.cpp (см. [testing.md](testing.md))

## Скачать тестовые модели

```bash
mkdir -p models

# Qwen3 (основная golden-модель)
curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Llama 3.2 1B Instruct
curl -L -o models/Llama-3.2-1B-Instruct-Q8_0.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Mistral 7B Instruct v0.2
curl -L -o models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```
