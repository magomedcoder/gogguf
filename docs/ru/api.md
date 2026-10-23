# HTTP API

Сервер запускается через `gogguf serve` (см. [CLI](cli.md)).

## Auth и rate limit

| Флаг           | По умолчанию | Описание                                                |
|----------------|--------------|---------------------------------------------------------|
| `--api-key`    | пусто        | требовать `Authorization: Bearer <key>` или `X-API-Key` |
| `--rate-limit` | `0`          | макс. запросов в минуту на IP (`0` = выкл.)             |

```bash
./build/gogguf serve -m model.gguf --api-key secret --rate-limit 60

curl -s 127.0.0.1:8000/v1/models -H "Authorization: Bearer secret"
curl -s 127.0.0.1:8000/v1/models -H "X-API-Key: secret"
```

При ошибке: `401` (`authentication_error`) или `429` (`rate_limit_error`) с телом `{"error":{"message":"...","type":"..."}}`.

В каждом ответе есть `X-Request-Id` (эхо заголовка запроса или сгенерированный id). При `--rate-limit` также: `X-RateLimit-Limit-Requests`, `X-RateLimit-Remaining-Requests`, `X-RateLimit-Reset-Requests`.

## Сессии и конкурентность

У модели один общий KV-cache; параллельная генерация сериализуется. Для multi-turn истории:

| Ключ сессии                             | Поведение                                       |
|-----------------------------------------|-------------------------------------------------|
| Заголовок `X-Session-Id`                | приоритет; Conversation в map сервера           |
| Поле `user` в JSON (если заголовка нет) | ключ `user:<id>`                                |
| Ни того ни другого                      | новый Conversation на каждый запрос (stateless) |

Embeddings не делают `Reset()` чужого chat KV: после embeddings следующий chat/completions при необходимости перезагружает KV из токенов сессии (`LoadKVCache`).

## Эндпоинты

| Метод | Путь                   | Описание                                |
|-------|------------------------|-----------------------------------------|
| GET   | `/v1/models`           | список загруженных моделей              |
| GET   | `/v1/models/{model}`   | одна модель                             |
| POST  | `/v1/chat/completions` | chat completions (messages + stream)    |
| POST  | `/v1/completions`      | legacy completions (prompt -> generate) |
| POST  | `/v1/embeddings`       | embeddings (hidden последнего токена)   |

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-4B",
      "object": "model",
      "created": 1710000000,
      "owned_by": "gogguf"
    }
  ]
}
```

## `GET /v1/models/{model}`

Тот же объект, что в списке, либо `404`, если id не совпадает с загруженной моделью.

## `POST /v1/chat/completions`

`Content-Type: application/json`

| Поле                    | Тип           | По умолчанию | Описание                                       |
|-------------------------|---------------|--------------|------------------------------------------------|
| `messages`              | array         | -            | `{role, content}` (обязательно)                |
| `model`                 | string        | имя из GGUF  | идентификатор модели                           |
| `max_tokens`            | int           | `128`        | максимум новых токенов                         |
| `max_completion_tokens` | int           | -            | алиас `max_tokens` (предпочтителен в OpenAI)   |
| `temperature`           | float         | `0`          | `0` = greedy                                   |
| `top_p`                 | float         | `1`          | nucleus sampling                               |
| `frequency_penalty`     | float         | `0`          | при `> 0` мапится во внутренний repeat penalty |
| `presence_penalty`      | float         | `0`          | OpenAI presence penalty (−2...2)               |
| `n`                     | int           | `1`          | только `1` (`n > 1` -> 400)                    |
| `stop`                  | string/array  | -            | стоп-последовательности                        |
| `stream`                | bool          | `false`      | SSE (`data: [DONE]` в конце)                   |
| `stream_options`        | object        | -            | `{ "include_usage": true }`                    |
| `seed`                  | int           | -            | seed sampler RNG                               |
| `user`                  | string        | -            | ключ сессии без `X-Session-Id`                 |
| `logit_bias`            | object        | -            | `{"token_id": bias, ...}`                      |
| `response_format`       | object        | -            | `text` / `json_object` (`json_schema` -> 400)  |
| `logprobs`              | -             | -            | **заглушка**                                   |
| `tools`                 | array         | -            | описания инструментов                          |
| `tool_choice`           | string/object | `auto`       | `auto` / `none` / `required` / по имени        |
| `parallel_tool_calls`   | bool          | `false`      | несколько tool_calls за один ход               |

`content` - строка или массив `{type:"text", text:"..."}`. Не-text части (например `image_url`) -> **400**.

В messages поддерживаются `tool_calls` (assistant) и `tool_call_id` / `name` (role `tool`).

Ответ без stream:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "Qwen3-4B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

Streaming: SSE-чанки, `finish_reason` в последнем чанке, затем `data: [DONE]`.

```bash
curl -s 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет"}],"max_tokens":32}'

curl -N 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет"}],"max_tokens":32,"stream":true}'
```

## `POST /v1/completions`

Legacy Completions API: `prompt` (строка или массив строк) -> генерация. Поддерживаются `stream`, `max_tokens`, sampling, `stop`, `echo`. Для chat templates предпочтительно `/v1/chat/completions`.

```bash
curl -s 127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Переведи фразу Привет мир на английский:","max_tokens":32}'
```

## `POST /v1/embeddings`

Hidden state последнего токена после output norm (до lm_head). Это генеративные GGUF-модели - не отдельные embedding-модели.

| Поле              | Тип    | По умолчанию | Описание                                    |
|-------------------|--------|--------------|---------------------------------------------|
| `input`           | any    | -            | строка, `[]string`, `[]int` или `[][]int`   |
| `model`           | string | GGUF         | id модели                                   |
| `encoding_format` | string | `float`      | `float` (JSON-массив) или `base64` (LE f32) |

```bash
curl 127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"Привет мир"}'
```
