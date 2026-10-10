# Использование как библиотеки

## Inference

```go
import "github.com/magomedcoder/gogguf"

engine, err := gogguf.Load("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 0, // matmul N слоёв на GPU; <= block_count, нужна CUDA-сборка
})
defer engine.Close() // освободить VRAM / backend при NGL > 0

ctx, err := engine.NewContext()

prompt, err := gogguf.FormatChatUser("Привет", gogguf.ChatOptions{
	Metadata: engine.Metadata(),
})
text, err := ctx.Generate(prompt, gogguf.GenerateParams{
	MaxTokens: 128,
	Sampler:   gogguf.Greedy,
})
```

### `LoadOptions` (важные поля)

| Поле                       | Смысл                                                                  |
|----------------------------|------------------------------------------------------------------------|
| `NGL`                      | число слоёв на GPU (`-ngl`)                                            |
| `NBatch`                   | chunk prefill (`-b`); `>1` эффективно только для Qwen3 dense; MoE -> 1 |
| `GPUMaxSeq`                | макс. длина GPU KV (`0` = auto, до 4096)                               |
| `GPUDevice` / `GPUDevices` | ordinal(s) после `CUDA_VISIBLE_DEVICES`                                |
| `TensorSplit`              | пропорции слоёв при multi-GPU                                          |
| `GPU`                      | готовый backend; иначе при `NGL > 0` откроется CUDA                    |

Где GPU доступен: [platforms.md](platforms.md).

## Пошаговый decode

```go
sess, err := ctx.StartGeneration(prompt)
for i := 0; i < maxTokens; i++ {
	id, err := sess.DecodeStep(gogguf.Greedy)
	if id < 0 {
		break
	}

	fmt.Print(sess.DecodeToken(id))
}
```

## Sampling с temperature / top-k / top-p / min-p

```go
sampler := gogguf.NewSampler(gogguf.SamplerConfig{
	Temp: 0.7,
	TopK: 40,
	TopP: 0.9,
	MinP: 0.05,
	Seed: 42,
})
text, err := ctx.Generate(prompt, gogguf.GenerateParams{
	MaxTokens:     64,
	Sampler:       sampler,
	RepeatPenalty: 1.1,
	RepeatLastN:   64,
})
```

## Загрузка через mmap (zero-copy веса)

Предпочтительный путь для больших файлов (как у CLI `run`/`serve`):

```go
engine, err := gogguf.LoadMapped("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{
	NGL:    0,
	NBatch: 1,
})
defer engine.Close()
```

Windows mmap / лимиты GPU: [platforms.md](platforms.md).

## GPU offload из кода

Нужна CUDA-сборка на Linux - см. [platforms.md](platforms.md).

```go
engine, err := gogguf.LoadMapped("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 28,
})
defer engine.Close()

devs, split, err := gogguf.ParseGPUDevices("0,1", "0.6,0.4")
engine, err = gogguf.LoadMapped(path, gogguf.LoadOptions{
	NGL:         28,
	GPUDevices:  devs,
	TensorSplit: split,
})
```

## Парсинг GGUF без inference

```go
import "github.com/magomedcoder/gogguf"

r, err := gogguf.OpenFile("./models/Qwen3-4B-Q8_0.gguf")

arch, _ := r.Metadata.String("general.architecture")
```

## HTTP-сервер из кода

```go
import (
	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/server"
)

engine, _ := gogguf.LoadMapped("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{})
defer engine.Close()

srv := server.New(engine, "./models/Qwen3-4B-Q8_0.gguf")

srv.ListenAndServe("127.0.0.1:8000")
```

Сессии и эндпоинты - как у CLI serve: см. [api.md](api.md).
