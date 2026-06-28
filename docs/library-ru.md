# Использование как библиотеки

[English version](library.md)

## Inference

```go
import "github.com/magomedcoder/gguf.go"

engine, err := gguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gguf.LoadOptions{
	NGL: 0, // matmul N слоёв на GPU; <= block_count, нужна CUDA-сборка
})
ctx, err := engine.NewContext()

prompt, err := gguf.FormatChatUser("Привет", gguf.ChatOptions{
	Metadata: engine.Metadata(),
})
text, err := ctx.Generate(prompt, gguf.GenerateParams{
	MaxTokens: 128,
	Sampler:   gguf.Greedy,
})
```

## Пошаговый decode

```go
sess, err := ctx.StartGeneration(prompt)
for i := 0; i < maxTokens; i++ {
	id, err := sess.DecodeStep(gguf.Greedy)
	if id < 0 {
		break
	}

	fmt.Print(sess.DecodeToken(id))
}
```

## Sampling с temperature / top-k / top-p / min-p

```go
sampler := gguf.NewSampler(gguf.SamplerConfig{
	Temp: 0.7,
	TopK: 40,
	TopP: 0.9,
	MinP: 0.05,
	Seed: 42,
})
text, err := ctx.Generate(prompt, gguf.GenerateParams{
	MaxTokens:     64,
	Sampler:       sampler,
	RepeatPenalty: 1.1,
	RepeatLastN:   64,
})
```

## Загрузка через mmap (zero-copy веса)

```go
engine, err := gguf.LoadMapped("./models/Qwen3-0.6B-Q8_0.gguf", gguf.LoadOptions{
	NGL: 0,
})
```

## GPU offload из кода

```go
engine, err := gguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gguf.LoadOptions{
	NGL: 28, // нужна сборка -tags cuda
})
```

## Парсинг GGUF без inference

```go
import "github.com/magomedcoder/gguf.go"

r, err := gguf.OpenFile("./models/Qwen3-0.6B-Q8_0.gguf")

arch, _ := r.Metadata.String("general.architecture")
```

## HTTP-сервер из кода

```go
import (
	"github.com/magomedcoder/gguf.go"
	"github.com/magomedcoder/gguf.go/server"
)

engine, _ := gguf.Load("./models/Qwen3-0.6B-Q8_0.gguf", gguf.LoadOptions{})

srv := server.New(engine, "./models/Qwen3-0.6B-Q8_0.gguf")

srv.ListenAndServe("127.0.0.1:8000")
```
