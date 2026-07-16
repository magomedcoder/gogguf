# HTTP API

[English version](api.md)

Сервер запускается через `gguf serve` (см. [CLI](cli-ru.md)).

## Эндпоинты

| Метод | Путь                   | Описание                              |
|-------|------------------------|---------------------------------------|
| GET   | `/v1/health`           | проверка состояния сервера            |
| GET   | `/v1/models`           | список моделей                        |
| POST  | `/v1/reset`            | сброс KV-cache на сервере (новый чат) |
| POST  | `/v1/chat/completions` | chat API (messages + stream)          |
| POST  | `/v1/embeddings`       | embeddings (пока не поддерживается)   |

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-0.6B",
      "object": "model"
    }
  ]
}
```

## `POST /v1/reset`

Сбрасывает KV-cache на сервере для multi-turn чата.

```bash
curl -s -X POST 127.0.0.1:8000/v1/reset
```

Ответ: `{"status":"ok"}`

## `POST /v1/chat/completions`

`Content-Type: application/json`

| Поле                           | Тип           | По умолчанию | Описание                                |
|--------------------------------|---------------|--------------|-----------------------------------------|
| `messages`                     | array         | -            | `{role, content}` (обязательно)         |
| `model`                        | string        | имя из GGUF  | идентификатор модели                    |
| `max_tokens`                   | int           | `128`        | максимум новых токенов                  |
| `temperature`                  | float         | `0`          | `0` = greedy                            |
| `top_k`                        | int           | `0`          | top-k sampling                          |
| `top_p`                        | float         | `1`          | nucleus sampling                        |
| `min_p`                        | float         | `0`          | min-p sampling                          |
| `repeat_penalty`               | float         | `1`          | штраф за повтор (`1` = выключено)       |
| `repeat_last_n`                | int           | `64`         | окно repeat-penalty                     |
| `stop`                         | string[]      | -            | стоп-последовательности                 |
| `stream`                       | bool          | `false`      | SSE (`data: [DONE]` в конце)            |
| `thinking` / `enable_thinking` | bool          | `false`      | режим размышления Qwen3                 |
| `tools`                        | array         | -            | описания инструментов (OpenAI-стиль)    |
| `tool_choice`                  | string/object | `auto`       | `auto` / `none` / `required` / по имени |
| `parallel_tool_calls`          | bool          | `false`      | несколько tool_calls за один ход        |

`content` - строка или массив `{type:"text", text:"..."}`.

В messages поддерживаются `tool_calls` (assistant) и `tool_call_id` / `name` (role `tool`).

Ответ без stream:

```json
{
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"Moscow\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

Если модель не вызывает инструмент, `finish_reason` = `"stop"`, поле `tool_calls` отсутствует.
Streaming - SSE: `data: {"choices":[{"delta":{"content":"..."}}]}` и `data: [DONE]`.

Примеры:

```bash
curl -s 127.0.0.1:8000/v1/models

curl -s 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет"}],"max_tokens":32}'

curl -N 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет"}],"max_tokens":32,"stream":true}'
```

## `POST /v1/embeddings`

Возвращает HTTP `501` - generative GGUF модели пока не поддерживают embeddings.
