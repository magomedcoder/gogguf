# GoGGUF

**GoGGUF** - лёгковесный способ запуска GGUF-моделей на Go.

> **GPU в активной разработке.**

Проект можно использовать как **библиотеку** (inference из Go-кода) и как **HTTP-сервер** (`gogguf serve` или пакет `server`).

**Внешних Go-зависимостей нет** - только стандартная библиотека.

**GPU / ОС:** см. [docs/ru/platforms.md](docs/ru/platforms.md).

## Что уже работает

- парсинг GGUF v2/v3 (`info`, `inspect`), memory-map (`LoadMapped`, zero-copy `RawView`);
- кванты на CPU и fused CUDA: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K;
- ops: RoPE, RMSNorm, GQA / flash-attn, SwiGLU, GeGLU, MLA (DeepSeek2);
- **SIMD:** AVX2 (amd64), NEON (arm64);
- forward + KV-cache: **Qwen3** (dense и MoE), **Qwen2**, **Llama 2/3**, **Mistral / Mixtral**, **Phi-2/3**, **Gemma / Gemma2**, **DeepSeek / DeepSeek2**;
- tokenizer BPE из метаданных GGUF;
- chat template ChatML/Qwen и Jinja (`--chat`, `--thinking`, tools);
- генерация: `gogguf run` (greedy / temperature / top-k / top-p / min-p / repeat penalty), prefill `-b` (по умолчанию 32 для dense);
- HTTP API: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (JSON + SSE); `seed`, `finish_reason` (`stop` / `length` / `tool_calls`), разбор tool calls (Qwen `<tool_call>` и Llama JSON);
- **CUDA** (`-tags cuda`, `-ngl N`): residency слоёв, GPU KV, batched GEMM, multi-GPU через P2P. Сам флаг `-tags cuda` модель на GPU не кладёт - нужен `-ngl`.

## Модели

**Работает:** Qwen3, Qwen2, Llama 2/3, Mistral, Mixtral, Phi-2/3, Gemma/Gemma2, DeepSeek/DeepSeek2  
**Скоро:** Vision, отдельные embedding-модели

Подробнее: [docs/ru/models.md](docs/ru/models.md).

## Быстрый старт

Пример модели для запуска CLI: `Qwen3-4B-Q8_0.gguf`

```bash
# скачать модель
mkdir -p models
curl -L -o models/Qwen3-4B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf

# собрать CLI
go build -o build/gogguf ./cmd/gogguf

# чат в терминале
./build/gogguf run -m ./models/Qwen3-4B-Q8_0.gguf --chat -p "Привет" -n 64

# или HTTP API (совместимый с OpenAI)
./build/gogguf serve -m ./models/Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000
```

На macOS те же команды (`darwin/arm64` или `amd64`). Windows (PowerShell):

```powershell
go build -o build/gogguf.exe ./cmd/gogguf
.\build\gogguf.exe run -m .\models\Qwen3-4B-Q8_0.gguf --chat -p "Привет" -n 64
.\build\gogguf.exe serve -m .\models\Qwen3-4B-Q8_0.gguf --host 127.0.0.1:8000
```

GPU (`-ngl`, Windows `nvcuda.dll`, macOS Metal): [docs/ru/platforms.md](docs/ru/platforms.md).

Опциональный простой чат-UI ([web-ui](web-ui/)): оставьте `serve` запущенным и в другом терминале:

```bash
cd web-ui
yarn install # или npm install
yarn dev # http://localhost:5173 - проксирует /api на :8000
```

Подробнее про API: [docs/ru/api.md](docs/ru/api.md). Про UI: [docs/ru/web-ui.md](docs/ru/web-ui.md).

## Документация

* [Платформы и видеокарты (где запускать)](docs/ru/platforms.md)
* [Сборка: CPU / Docker](docs/ru/build.md)
* [CLI: `info`, `inspect`, `run`, `serve`](docs/ru/cli.md)
* [HTTP API сервера](docs/ru/api.md)
* [Простой web UI](docs/ru/web-ui.md)
* [Inference из Go-кода](docs/ru/library.md)
* [`debugtok`, `vocab`, `bench`](docs/ru/tools.md)
* [Формат GGUF](docs/ru/GGUF-FORMAT.md)
* [Поддерживаемые и планируемые модели](docs/ru/models.md)
* [Тестирование](docs/ru/testing.md)

## Участие в разработке

Мы рады любому вкладу! Подробности - в [CONTRIBUTING.md](CONTRIBUTING.md)

---

[English version](README.md)
