# Поддерживаемые модели

[English version](models.md)

## Работает

| Архитектура | Статус    | Модели                          |
|-------------|-----------|---------------------------------|
| Qwen3       | проверено | Qwen3-0.6B, Qwen3-8B, Qwen3-14B |
| Llama 3     | базово    | Llama-3.2-1B                    |

Форматы весов: **Q8_0**, **Q4_0**, **Q4_K**.

```bash
mkdir -p models

curl -L -o models/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

curl -L -o models/Qwen3-8B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf

curl -L -o models/Qwen3-14B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q8_0.gguf

curl -L -o models/Llama-3.2-1B-Instruct-Q8_0.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
```

```bash
./build/gogguf run -m models/Qwen3-0.6B-Q8_0.gguf --chat -p "Hello" -n 64

./build/gogguf run -m models/Llama-3.2-1B-Instruct-Q8_0.gguf --chat -p "Hello" -n 32
```

## Скоро

1. Llama 2
2. Mistral
3. Phi
4. Gemma
