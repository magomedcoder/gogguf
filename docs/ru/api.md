# HTTP API

Сервер запускается через `gogguf serve` (см. [CLI](cli.md)).

## Auth и rate limit

| Флаг           | По умолчанию | Описание                                                 |
|----------------|--------------|----------------------------------------------------------|
| `--api-key`    | пусто        | требовать `Authorization: Bearer <key>` или `X-API-Key`  |
| `--rate-limit` | `0`          | макс. запросов в минуту на IP (`0` = выкл.)              |

```bash
./build/gogguf serve -m model.gguf --api-key secret --rate-limit 60

curl -s 127.0.0.1:8000/v1/models -H "Authorization: Bearer secret"
curl -s 127.0.0.1:8000/v1/models -H "X-API-Key: secret"
```

Ошибки: `401` (`authentication_error`) или `429` (`rate_limit_error`) - `{"error":{"message":"...","type":"..."}}`.

## Эндпоинты

| Метод | Путь                   | Описание                                |
|-------|------------------------|-----------------------------------------|
| GET   | `/v1/models`           | список моделей                          |
| GET   | `/v1/models/{model}`   | одна модель                             |
| POST  | `/v1/chat/completions` | chat completions (messages + stream)    |
| POST  | `/v1/completions`      | legacy completions (prompt -> generate) |
| POST  | `/v1/embeddings`       | embeddings (last-token hidden)          |

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

| Поле                                                            | Тип           | По умолчанию | Описание                                       |
|-----------------------------------------------------------------|---------------|--------------|------------------------------------------------|
| `messages`                                                      | array         | -            | `{role, content}` (обязательно)                |
| `model`                                                         | string        | имя из GGUF  | идентификатор модели                           |
| `max_tokens`                                                    | int           | `128`        | максимум новых токенов                         |
| `max_completion_tokens`                                         | int           | -            | алиас `max_tokens` (предпочтителен в OpenAI)   |
| `temperature`                                                   | float         | `0`          | `0` = greedy                                   |
| `top_p`                                                         | float         | `1`          | nucleus sampling                               |
| `frequency_penalty`                                             | float         | `0`          | при `> 0` мапится во внутренний repeat penalty |
| `presence_penalty`                                              | float         | `0`          | **заглушка** (принимается, игнорируется)       |
| `n`                                                             | int           | `1`          | поддерживается только `1`                      |
| `stop`                                                          | string/array  | -            | стоп-последовательности                        |
| `stream`                                                        | bool          | `false`      | SSE (`data: [DONE]` в конце)                   |
| `stream_options`                                                | object        | -            | `{ "include_usage": true }` для usage в stream |
| `seed` / `user` / `logit_bias` / `logprobs` / `response_format` | -             | -            | **заглушки**                                   |
| `tools`                                                         | array         | -            | описания инструментов (OpenAI-стиль)           |
| `tool_choice`                                                   | string/object | `auto`       | `auto` / `none` / `required` / по имени        |
| `parallel_tool_calls`                                           | bool          | `false`      | несколько tool_calls за один ход               |

`content` - строка или массив `{type:"text", text:"..."}`.

В messages поддерживаются `tool_calls` (assistant) и `tool_call_id` / `name` (role `tool`).

Streaming: SSE-чанки с `finish_reason` в последнем чанке, затем `data: [DONE]`.

```bash
curl -s 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет"}],"max_tokens":32}'
```

## `POST /v1/completions`

Legacy Completions API: `prompt` (строка или массив строк) -> генерация. Поддерживаются `stream`, `max_tokens`, sampling, `stop`, `echo`. Для chat templates предпочтительно `/v1/chat/completions`.

```bash
curl -s 127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":32}'
```

## `POST /v1/embeddings`

Hidden state последнего токена после output norm (до lm_head). Генеративные GGUF-модели - не отдельные embedding-модели.

`input`: строка, массив строк, token ids (`[]int`) или батчи (`[][]int`).

```bash
curl 127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world"}'
```
