# GoGGUF

**GoGGUF** - лёгковесный способ запуска GGUF-моделей на Go без llama.cpp.

> **Проект в активной разработке.** Поддержка GPU (CUDA) пока на раннем этапе.

Проект можно использовать как **библиотеку** (inference из Go-кода) и как **HTTP-сервер** (`gguf serve` или
пакет `server`).

**Внешних Go-зависимостей нет** - только стандартная библиотека.

Опционально: **CUDA** через Driver API (`libcuda.so`, сборка `-tags cuda`, CGO).

## Что уже работает

- парсинг GGUF v2/v3 (`info`, `inspect`), memory-map (`LoadMapped`, zero-copy `RawView`);
- деквантизация и matmul: Q8_0, Q4_0, Q4_K, Q5_K, Q6_K;
- базовые ops: RoPE, RMSNorm, GQA attention, SwiGLU;
- **SIMD** matmul FP32: AVX2 (amd64), NEON (arm64); Q8_0 dot: AVX2 (amd64);
- forward pass **Qwen3**, **Llama 2/3**, **Mistral**, **Phi-2/3**, **Gemma/Gemma2** + KV-cache;
- tokenizer BPE из метаданных GGUF;
- chat template ChatML/Qwen и Jinja (`--chat`, `--thinking`, `FormatChatUser`);
- генерация текста: `gguf run` (greedy / temperature / top-k / top-p / min-p / repeat penalty);
- HTTP-сервер: `gguf serve` (`/v1/models`, `/v1/chat/completions`, JSON + SSE);
- **CUDA offload** (`-ngl N`): matmul первых N transformer-слоёв на GPU (сборка `-tags cuda`).

## Модели

**Работает:** Qwen3, Llama 2/3, Mistral, Phi-2/3, Gemma/Gemma2
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

Опциональный простой чат-UI ([web-ui](web-ui/)): оставьте `serve` запущенным и в другом терминале:

```bash
cd web-ui
yarn install   # или npm install
yarn dev       # http://localhost:5173 - проксирует /api на :8000
```

Подробнее про API: [docs/ru/api.md](docs/ru/api.md). Про UI: [web-ui/README-ru.md](web-ui/README-ru.md).

## Документация

* [Сборка: CPU, CUDA, Docker](docs/ru/build.md)
* [CLI: `info`, `inspect`, `run`, `serve`](docs/ru/cli.md)
* [HTTP API сервера](docs/ru/api.md)
* [Простой web UI](web-ui/README-ru.md)
* [Inference из Go-кода](docs/ru/library.md)
* [`debugtok`, `vocab`, `bench`](docs/ru/tools.md)
* [Формат GGUF](docs/ru/GGUF-FORMAT.md)
* [Поддерживаемые и планируемые модели](docs/ru/models.md)
* [Тестирование](docs/ru/testing.md)

## Участие в разработке

Мы рады любому вкладу! Подробности - в [CONTRIBUTING.md](CONTRIBUTING.md)

---

[English version](README.md)
