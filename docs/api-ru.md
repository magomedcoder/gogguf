# HTTP API

[English version](api.md)

Сервер запускается через `gguf serve` (см. [CLI](cli-ru.md)).

## Эндпоинты

| Метод | Путь           | Описание                              |
|-------|----------------|---------------------------------------|
| GET   | `/health`      | проверка состояния сервера            |
| GET   | `/models`      | метаданные загруженной модели         |
| POST  | `/reset`       | сброс KV-cache на сервере (новый чат) |
| POST  | `/generate`    | генерация текста (JSON или SSE)       |
| POST  | `/completions` | chat API (messages + stream)          |

## `POST /generate`

`Content-Type: application/json`

| Поле             | Тип    | По умолчанию | Описание                    |
|------------------|--------|--------------|-----------------------------|
| `prompt`         | string | -            | текст запроса (обязательно) |
| `max_tokens`     | int    | `128`        | максимум новых токенов      |
| `temperature`    | float  | `0`          | `0` = greedy                |
| `top_k`          | int    | `0`          | top-k sampling              |
| `top_p`          | float  | `1`          | nucleus sampling            |
| `min_p`          | float  | `0`          | min-p sampling              |
| `repeat_penalty` | float  | `1`          | штраф за повтор (`1` = off) |
| `repeat_last_n`  | int    | `0`          | окно истории (0 = 64)       |
| `seed`           | uint   | `0`          | seed PRNG                   |
| `chat`           | bool   | `false`      | ChatML/Qwen template        |
| `stream`         | bool   | `false`      | SSE streaming               |
| `system`         | string | -            | system prompt (с `chat`)    |
| `thinking`       | bool   | `false`      | режим размышления Qwen3     |

Ответ без stream:

```json
{
  "text": "...",
  "tokens": 32
}
```

Streaming (SSE) - события `data: {"token":"..."}` и в конце `data: {"done":true}`.

Примеры:

```bash
curl -s 127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Привет","chat":true,"max_tokens":32}'

curl -N 127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Привет","chat":true,"stream":true,"max_tokens":32}'

curl -s 127.0.0.1:8000/models
```

## `POST /reset`

Сбрасывает KV-cache на сервере для multi-turn чата. 

```bash
curl -s -X POST 127.0.0.1:8000/reset
```

Ответ: `{"status":"ok"}`

## `POST /completions`

Chat API с массивом `messages`.

```bash
curl -s 127.0.0.1:8000/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет"}],"max_tokens":32}'
```

Пример ответа:

```json
{
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

Поддерживаются `temperature`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `stream`, `thinking`.
