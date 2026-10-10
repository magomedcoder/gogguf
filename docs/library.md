# Library usage

## Inference

```go
import "github.com/magomedcoder/gogguf"

engine, err := gogguf.Load("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 0, // matmul N layers on GPU; <= block_count, CUDA build required
})
defer engine.Close() // free VRAM / backend when NGL > 0

ctx, err := engine.NewContext()

prompt, err := gogguf.FormatChatUser("Hello", gogguf.ChatOptions{
	Metadata: engine.Metadata(),
})
text, err := ctx.Generate(prompt, gogguf.GenerateParams{
	MaxTokens: 128,
	Sampler:   gogguf.Greedy,
})
```

### `LoadOptions` (important fields)

| Field                      | Meaning                                                                    |
|----------------------------|----------------------------------------------------------------------------|
| `NGL`                      | layers on GPU (`-ngl`)                                                     |
| `NBatch`                   | prefill chunk (`-b`); `>1` effective for Qwen3 dense only; MoE forced to 1 |
| `GPUMaxSeq`                | max GPU KV length (`0` = auto, up to 4096)                                 |
| `GPUDevice` / `GPUDevices` | ordinal(s) after `CUDA_VISIBLE_DEVICES`                                    |
| `TensorSplit`              | layer proportions for multi-GPU                                            |
| `GPU`                      | ready backend; otherwise CUDA opens when `NGL > 0`                         |

Which OS support GPU: [platforms.md](platforms.md).

## Step-by-step decode

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

## Sampling with temperature / top-k / top-p / min-p

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

## Load via mmap (zero-copy weights)

Preferred for large files (same path as CLI `run`/`serve`):

```go
engine, err := gogguf.LoadMapped("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{
	NGL:    0,
	NBatch: 1,
})
defer engine.Close()
```

Windows mmap / GPU launch limits: [platforms.md](platforms.md).

## GPU offload from code

Requires a CUDA build on Linux - see [platforms.md](platforms.md).

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

## Parse GGUF without inference

```go
import "github.com/magomedcoder/gogguf"

r, err := gogguf.OpenFile("./models/Qwen3-4B-Q8_0.gguf")

arch, _ := r.Metadata.String("general.architecture")
```

## HTTP server from code

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

Sessions and endpoints match CLI serve - see [api.md](api.md).
