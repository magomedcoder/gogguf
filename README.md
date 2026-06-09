# gguf.go - запуск ML-моделей в формате **GGUF** на чистом **Go**.

> **Ранний этап разработки.**

Формат **GGUF** используется в экосистеме llama.cpp. **gguf.go** - лёгковесный способ запуска GGUF-моделей на языке Go без использования llama.cpp.

---

## Что уже работает

- парсинг GGUF v2/v3 (`info`, `inspect`);
- деквантизация Q8_0, загрузка весов (`quant`, `tensor`, `weights`);
- базовые ops: RoPE, RMSNorm, GQA attention, SwiGLU;
- forward pass Qwen3 + KV-cache (`model/qwen3`, `runtime`).

Генерация текста (inference) пока не реализована.

---

```
https://huggingface.co/Qwen/Qwen3-0.6B-GGUF?show_file_info=Qwen3-0.6B-Q8_0.gguf
```

```bash
go build -o build/gguf ./cmd/gguf

# Краткая информация о модели
./build/gguf info -m ./models/Qwen3-0.6B-Q8_0.gguf
# Просмотр метаданных и тензоров
./build/gguf inspect ./models/Qwen3-0.6B-Q8_0.gguf
```
