# HTTP API

Start the server with `gogguf serve` (see [CLI](cli.md)).


## Auth and rate limit

| Flag           | Default | Description                                              |
|----------------|---------|----------------------------------------------------------|
| `--api-key`    | empty   | require `Authorization: Bearer <key>` or `X-API-Key`     |
| `--rate-limit` | `0`     | max requests per minute per IP (`0` = off)               |

```bash
./build/gogguf serve -m model.gguf --api-key secret --rate-limit 60

curl -s 127.0.0.1:8000/v1/models -H "Authorization: Bearer secret"
curl -s 127.0.0.1:8000/v1/models -H "X-API-Key: secret"
```

On failure: `401` (`authentication_error`) or `429` (`rate_limit_error`) with `{"error":{"message":"...","type":"..."}}`.

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
      "owned_by": "gogguf"
    }
  ]
}
```

## `GET /v1/models/{model}`

Same object as in the list, or `404` if the id does not match the loaded model.

## `POST /v1/chat/completions`

`Content-Type: application/json`

| Field                                                           | Type          | Default   | Description                                     |
|-----------------------------------------------------------------|---------------|-----------|-------------------------------------------------|
| `messages`                                                      | array         | -         | `{role, content}` (required)                    |
| `model`                                                         | string        | GGUF name | model id                                        |
| `max_tokens`                                                    | int           | `128`     | max new tokens                                  |
| `max_completion_tokens`                                         | int           | -         | alias for `max_tokens` (preferred by OpenAI)    |
| `temperature`                                                   | float         | `0`       | `0` = greedy                                    |
| `top_p`                                                         | float         | `1`       | nucleus sampling                                |
| `frequency_penalty`                                             | float         | `0`       | mapped to internal repeat penalty when `> 0`    |
| `presence_penalty`                                              | float         | `0`       | **stub** (accepted, ignored)                    |
| `n`                                                             | int           | `1`       | only `1` supported                              |
| `stop`                                                          | string/array  | -         | stop sequences                                  |
| `stream`                                                        | bool          | `false`   | SSE streaming (`data: [DONE]` at end)           |
| `stream_options`                                                | object        | -         | `{ "include_usage": true }` for usage in stream |
| `seed` / `user` / `logit_bias` / `logprobs` / `response_format` | -             | -         | **stubs** (accepted, ignored)                   |
| `tools`                                                         | array         | -         | OpenAI-style tool definitions                   |
| `tool_choice`                                                   | string/object | `auto`    | `auto` / `none` / `required` / named            |
| `parallel_tool_calls`                                           | bool          | `false`   | allow multiple tool calls in one turn           |

`content` may be a string or an array of `{type:"text", text:"..."}` parts.

Messages may include `tool_calls` (assistant) and `tool_call_id` / `name` (tool role).

Non-streaming response:

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

Request `input`: string, array of strings, token ids (`[]int`), or batches (`[][]int`).

```bash
curl 127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world"}'
```
