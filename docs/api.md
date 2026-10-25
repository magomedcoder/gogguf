# HTTP API

Start the server with `gogguf serve` (see [CLI](cli.md)).

## Auth and rate limit

| Flag           | Default | Description                                          |
|----------------|---------|------------------------------------------------------|
| `--api-key`    | empty   | require `Authorization: Bearer <key>` or `X-API-Key` |
| `--rate-limit` | `0`     | max requests per minute per IP (`0` = off)           |

```bash
./build/gogguf serve -m model.gguf --api-key secret --rate-limit 60

curl -s 127.0.0.1:8000/v1/models -H "Authorization: Bearer secret"
curl -s 127.0.0.1:8000/v1/models -H "X-API-Key: secret"
```

On failure: `401` (`authentication_error`) or `429` (`rate_limit_error`) with `{"error":{"message":"...","type":"..."}}`.

Every response includes `X-Request-Id` (echoes the request header or a generated id). With `--rate-limit`, responses also set `X-RateLimit-Limit-Requests`, `X-RateLimit-Remaining-Requests`, and `X-RateLimit-Reset-Requests`.

## Sessions and concurrency

The model has one shared KV-cache; concurrent generation is serialized. For multi-turn history:

| Session key                | Behavior                                       |
|----------------------------|------------------------------------------------|
| Header `X-Session-Id`      | preferred; Conversation kept in the server map |
| JSON `user` (if no header) | key `user:<id>`                                |
| Neither                    | new Conversation per request (stateless)       |

Embeddings do not `Reset()` another session’s chat KV; the next chat/completions may reload KV from that session’s tokens (`LoadKVCache`).

## Endpoints

| Method | Path                   | Description                             |
|--------|------------------------|-----------------------------------------|
| GET    | `/v1/models`           | list loaded models                      |
| GET    | `/v1/models/{model}`   | retrieve one model                      |
| POST   | `/v1/chat/completions` | chat completions (messages + stream)    |
| POST   | `/v1/completions`      | legacy completions (prompt -> generate) |
| POST   | `/v1/embeddings`       | last-token hidden embeddings            |

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-4B",
      "object": "model",
      "created": 1710000000,
      "owned_by": "gogguf",
      "architecture": "qwen3",
      "context_length": 40960,
      "embedding_length": 2560
    }
  ]
}
```

Extra fields come from GGUF metadata when present (`architecture`, `context_length`, `embedding_length`).

## `GET /v1/models/{model}`

Same object as in the list, or `404` if the id does not match the loaded model.

## `POST /v1/chat/completions`

`Content-Type: application/json`

| Field                   | Type          | Default   | Description                                                                |
|-------------------------|---------------|-----------|----------------------------------------------------------------------------|
| `messages`              | array         | -         | `{role, content}` (required)                                               |
| `model`                 | string        | GGUF name | model id                                                                   |
| `max_tokens`            | int           | `128`     | max new tokens                                                             |
| `max_completion_tokens` | int           | -         | alias for `max_tokens` (preferred by OpenAI)                               |
| `temperature`           | float         | `0`       | `0` = greedy                                                               |
| `top_p`                 | float         | `1`       | nucleus sampling                                                           |
| `frequency_penalty`     | float         | `0`       | mapped to internal repeat penalty when `> 0`                               |
| `presence_penalty`      | float         | `0`       | OpenAI-style presence penalty (−2...2)                                     |
| `n`                     | int           | `1`       | only `1` supported (`n > 1` -> 400)                                        |
| `stop`                  | string/array  | -         | stop sequences                                                             |
| `stream`                | bool          | `false`   | SSE streaming (`data: [DONE]` at end)                                      |
| `stream_options`        | object        | -         | `{ "include_usage": true }` for usage in stream                            |
| `seed`                  | int           | -         | sampler RNG seed                                                           |
| `user`                  | string        | -         | session key when `X-Session-Id` is absent                                  |
| `logit_bias`            | object        | -         | `{"token_id": bias, ...}` added to logits                                  |
| `response_format`       | object        | -         | `{ "type": "text" }` or `{ "type": "json_object" }` (`json_schema` -> 400) |
| `logprobs`              | bool          | `false`   | non-stream: per-token logprobs in `choices[0].logprobs`                    |
| `top_logprobs`          | int           | `0`       | `0..20` alternatives per token (requires `logprobs=true`)                  |
| `tools`                 | array         | -         | OpenAI-style tool definitions                                              |
| `tool_choice`           | string/object | `auto`    | `auto` / `none` / `required` / named                                       |
| `parallel_tool_calls`   | bool          | `false`   | allow multiple tool calls in one turn                                      |

`content` may be a string or an array of `{type:"text", text:"..."}` parts. Non-text parts (e.g. `image_url`) -> **400**.

Messages may include `tool_calls` (assistant) and `tool_call_id` / `name` (tool role).

Non-streaming response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "Qwen3-4B",
  "system_fingerprint": "fp_gogguf_...",
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

Streaming uses SSE chunks with `finish_reason` on the last chunk, then `data: [DONE]`.

```bash
curl -s 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

curl -N 127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

## `POST /v1/completions`

Legacy Completions API: `prompt` (string or string array) -> generation. Supports `stream`, `max_tokens`, sampling, `stop`, `echo`. Prefer `/v1/chat/completions` for chat templates.

```bash
curl -s 127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":32}'
```

## `POST /v1/embeddings`

Last-token hidden state after output norm (before lm_head). Generative GGUF models - not dedicated embedding models.

| Field             | Type   | Default | Description                                      |
|-------------------|--------|---------|--------------------------------------------------|
| `input`           | any    | -       | string, `[]string`, `[]int`, or `[][]int`        |
| `model`           | string | GGUF    | model id                                         |
| `encoding_format` | string | `float` | `float` (JSON array) or `base64` (LE float32)    |

```bash
curl 127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world"}'

curl 127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello","encoding_format":"base64"}'
```
