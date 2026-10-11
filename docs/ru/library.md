# Использование как библиотеки

Как встроить inference GGUF в свой Go-код.

Корневой пакет: `import "github.com/magomedcoder/gogguf"`.  
Для полного контроля над `model.Model` / backends импортируйте `pkg/runtime` и соседние пакеты напрямую.

**Конкурентность:** один in-flight Forward/Generate на `Engine` (общий KV). При шаринге между goroutine - сериализуйте.

**Sampler по умолчанию:** в library/CLI temperature `0` (greedy). OpenAI HTTP-клиенты часто шлют `temperature: 1` - см. [api.md](api.md).

## Inference

```go
import "github.com/magomedcoder/gogguf"

engine, err := gogguf.LoadMapped("./models/Qwen3-4B-Q8_0.gguf", gogguf.LoadOptions{
	NGL: 0, // слои на GPU; при > 0 нужна CUDA-сборка
})
defer engine.Close()

ctx, err := engine.NewContext()

prompt, err := gogguf.FormatChatUser("Привет", gogguf.ChatOptions{
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
	{Role: "user", Content: "Привет"},
	{Role: "assistant", Content: "Здравствуйте!"},
	{Role: "user", Content: "Как дела?"},
}
prompt, err := gogguf.FormatChatMessages(msgs, gogguf.ChatOptions{Metadata: engine.Metadata()})
conv, err := ctx.NewConversation()
text, err := conv.Generate(prompt, gogguf.GenerateParams{MaxTokens: 64, Sampler: gogguf.Greedy})
```

### Embeddings

```go
vec, tokens, err := engine.EmbedText("hello")
// или engine.EmbedTokens(tokenIDs)
```

### `LoadOptions`

| Поле                                                    | Смысл                                                                                   |
|---------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `NGL`                                                   | слои на GPU (`0` = CPU)                                                                 |
| `NBatch`                                                | chunk prefill; **`>1` только для Qwen3 dense** (другие arch игнор; MoE принудительно 1) |
| `GPUMaxSeq`                                             | макс. длина GPU KV (`0` = auto, до 4096)                                                |
| `GPUDevice` / `GPUDevices`                              | CUDA ordinal(s)                                                                         |
| `TensorSplit`                                           | пропорции слоёв для multi-GPU                                                           |
| `GPU`                                                   | готовый `device.Backend`; иначе CUDA открывается при `NGL > 0`                          |
| `HiddenResidency` / `QKVResidency` / `AttnFFNResidency` | `*bool` вкл/выкл fused GPU paths; `nil` = вкл.                                          |

Платформы GPU: [platforms.md](platforms.md). Hub env (`-hf`): [cli.md](cli.md#переменные-окружения-hugging-face).

## Загрузка из открытого reader

```go
r, closer, err := format.OpenFileOwned(path) // или свой *format.Reader
engine, err := gogguf.LoadReader(r, gogguf.LoadOptions{}, closer)
```

(`LoadReader` также есть в `pkg/runtime`.)

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

## Sampling

```go
sampler := gogguf.NewSampler(gogguf.SamplerConfig{
	Temp: 0.7, TopK: 40, TopP: 0.9, MinP: 0.05, Seed: 42,
})
```

## GPU offload

```go
devs, split, err := gogguf.ParseGPUDevices("0,1", "0.6,0.4") // хелпер как в CLI
engine, err = gogguf.LoadMapped(path, gogguf.LoadOptions{
	NGL: 28, GPUDevices: devs, TensorSplit: split,
})
```

## Парсинг GGUF без inference

```go
r, err := gogguf.OpenFile("./models/Qwen3-4B-Q8_0.gguf")
arch, _ := r.Metadata.String("general.architecture")
```

Enum dtype тензора: `gogguf.DType` / `pkg/dtype` (`Q8_0`, `F32`, ...).

## Опциональный HTTP (`pkg/server`)

Для library use не обязателен. Можно повесить на mux или слушать:

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
// или: srv.ListenAndServe("127.0.0.1:8000")
```
