# Library usage

How to run GGUF inference from your own Go code.

Root package: `import "github.com/magomedcoder/gogguf"`.  
For full control of `model.Model` / backends, import `pkg/runtime` and related packages directly.

**Concurrency:** one in-flight Forward/Generate per `Engine` (shared KV). Serialize if sharing across goroutines.

**Sampler defaults:** library/CLI default temperature `0` (greedy). OpenAI HTTP clients often send `temperature: 1` - see [api.md](api.md).

## Inference

```go
import "github.com/magomedcoder/gogguf"

engine, err := gogguf.LoadMapped("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 0, // layers on GPU; CUDA build required when > 0
})
defer engine.Close()

ctx, err := engine.NewContext()

prompt, err := gogguf.FormatChatUser("Hello", gogguf.ChatOptions{
	Metadata: engine.Metadata(),
})
text, err := ctx.Generate(prompt, gogguf.GenerateParams{
	MaxTokens: 128,
	Sampler:   gogguf.Greedy,
})
```

### Multi-turn

```go
msgs := []gogguf.ChatMessage{
	{Role: "user", Content: "Hi"},
	{Role: "assistant", Content: "Hello!"},
	{Role: "user", Content: "How are you?"},
}
prompt, err := gogguf.FormatChatMessages(msgs, gogguf.ChatOptions{Metadata: engine.Metadata()})
conv, err := ctx.NewConversation()
text, err := conv.Generate(prompt, gogguf.GenerateParams{MaxTokens: 64, Sampler: gogguf.Greedy})
```

### Embeddings

```go
vec, tokens, err := engine.EmbedText("hello")
// or engine.EmbedTokens(tokenIDs)
```

### `LoadOptions`

| Field                                                   | Meaning                                                                           |
|---------------------------------------------------------|-----------------------------------------------------------------------------------|
| `NGL`                                                   | layers on GPU (`0` = CPU)                                                         |
| `NBatch`                                                | prefill chunk; **`>1` only for Qwen3 dense** (other arch ignore; MoE forced to 1) |
| `GPUMaxSeq`                                             | max GPU KV length (`0` = auto, up to 4096)                                        |
| `GPUDevice` / `GPUDevices`                              | CUDA ordinal(s)                                                                   |
| `TensorSplit`                                           | layer proportions for multi-GPU                                                   |
| `GPU`                                                   | ready `device.Backend`; else CUDA opens when `NGL > 0`                            |
| `HiddenResidency` / `QKVResidency` / `AttnFFNResidency` | `*bool` toggles fused GPU paths; `nil` = enabled                                  |

GPU platforms: [platforms.md](platforms.md). Hub env (`-hf`): [cli.md](cli.md#environment-hugging-face).

## Load from an open reader

```go
r, closer, err := format.OpenFileOwned(path) // or build your own *format.Reader
engine, err := gogguf.LoadReader(r, gogguf.LoadOptions{}, closer)
```

(`LoadReader` is also on `pkg/runtime`.)

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

## Sampling

```go
sampler := gogguf.NewSampler(gogguf.SamplerConfig{
	Temp: 0.7, TopK: 40, TopP: 0.9, MinP: 0.05, Seed: 42,
})
```

## GPU offload

```go
devs, split, err := gogguf.ParseGPUDevices("0,1", "0.6,0.4") // CLI helper
engine, err = gogguf.LoadMapped(path, gogguf.LoadOptions{
	NGL: 28, GPUDevices: devs, TensorSplit: split,
})
```

## Parse GGUF without inference

```go
r, err := gogguf.OpenFile("./models/Qwen3-4B-Q8_0.gguf")
arch, _ := r.Metadata.String("general.architecture")
```

Tensor dtype enum: `gogguf.DType` / `pkg/dtype` (`Q8_0`, `F32`, ...).

## Optional HTTP (`pkg/server`)

Not required for library use. Mount into your mux or listen:

```go
import (
	"net/http"

	"github.com/magomedcoder/gogguf"
	"github.com/magomedcoder/gogguf/pkg/server"
)

engine, _ := gogguf.LoadMapped("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{})
defer engine.Close()

srv := server.New(engine, "", server.Options{ModelID: "my-model"})
http.Handle("/v1/", srv.Handler())
// or: srv.ListenAndServe("127.0.0.1:8000")
```
