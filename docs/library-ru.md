# Использование как библиотеки

[English version](library.md)

## Inference

```go
import "github.com/magomedcoder/gogguf"

engine, err := gogguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 0, // matmul N слоёв на GPU; <= block_count, нужна CUDA-сборка
})
ctx, err := engine.NewContext()

prompt, err := gogguf.FormatChatUser("Привет", gogguf.ChatOptions{
	Metadata: engine.Metadata(),
})
text, err := ctx.Generate(prompt, gogguf.GenerateParams{
	MaxTokens: 128,
	Sampler:   gogguf.Greedy,
})
```

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

```go
engine, err := gogguf.LoadMapped("./models/Qwen3-0.6B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 0,
})
```

## GPU offload из кода

```go
engine, err := gogguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 28, // нужна сборка -tags cuda
})
```

## Парсинг GGUF без inference

```go
import "github.com/magomedcoder/gogguf"

r, err := gogguf.OpenFile("./models/Qwen3-0.6B-Q8_0.gguf")

arch, _ := r.Metadata.String("general.architecture")
```

## HTTP-сервер из кода

```go
import (
	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/server"
)

engine, _ := gogguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gogguf.LoadOptions{})

srv := server.New(engine, "./models/Qwen3-0.6B-Q8_0.gguf")

srv.ListenAndServe("127.0.0.1:8000")
```
