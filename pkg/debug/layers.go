package debug

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

const layersMagic = "GGUFLYR1"

type LayersMeta struct {
	Magic     string    `json:"magic"`
	Model     string    `json:"model"`
	Prompt    string    `json:"prompt"`
	Chat      bool      `json:"chat,omitempty"`
	Tokens    []int     `json:"tokens"`
	Embd      int       `json:"embd"`
	NumLayers int       `json:"num_layers"`
	EmbedRMS  float32   `json:"embed_rms"`
	LayerRMS  []float32 `json:"layer_rms"`
	Greedy    int       `json:"greedy"`
	Backend   string    `json:"backend,omitempty"`
	NGL       int       `json:"ngl,omitempty"`
}

// LayersDump - embed + hidden после каждого слоя
type LayersDump struct {
	Embed  []float32
	Layers [][]float32 // [num_layers][embd]
}

// WriteLayersBin magic + embd + n_layers + embed + layers
func WriteLayersBin(w io.Writer, dump LayersDump) error {
	if len(dump.Embed) == 0 {
		return fmt.Errorf("debug: пустой embed")
	}
	embd := len(dump.Embed)
	for i, ly := range dump.Layers {
		if len(ly) != embd {
			return fmt.Errorf("debug: layer[%d] len=%d, embd=%d", i, len(ly), embd)
		}
	}

	if _, err := io.WriteString(w, layersMagic); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint32(embd)); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint32(len(dump.Layers))); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, dump.Embed); err != nil {
		return err
	}

	for _, ly := range dump.Layers {
		if err := binary.Write(w, binary.LittleEndian, ly); err != nil {
			return err
		}
	}

	return nil
}

// ReadLayersBin читает dump от WriteLayersBin
func ReadLayersBin(r io.Reader) (LayersDump, error) {
	magic := make([]byte, 8)
	if _, err := io.ReadFull(r, magic); err != nil {
		return LayersDump{}, err
	}

	if string(magic) != layersMagic {
		return LayersDump{}, fmt.Errorf("debug: неверный magic %q, ожидали %q", magic, layersMagic)
	}

	var embd, nLayers uint32
	if err := binary.Read(r, binary.LittleEndian, &embd); err != nil {
		return LayersDump{}, err
	}

	if err := binary.Read(r, binary.LittleEndian, &nLayers); err != nil {
		return LayersDump{}, err
	}

	dump := LayersDump{
		Embed:  make([]float32, embd),
		Layers: make([][]float32, nLayers),
	}
	if err := binary.Read(r, binary.LittleEndian, dump.Embed); err != nil {
		return LayersDump{}, err
	}

	for i := range dump.Layers {
		dump.Layers[i] = make([]float32, embd)
		if err := binary.Read(r, binary.LittleEndian, dump.Layers[i]); err != nil {
			return LayersDump{}, err
		}
	}

	return dump, nil
}

// SaveLayersDump пишет path.bin и path.json
func SaveLayersDump(path string, meta LayersMeta, dump LayersDump) error {
	meta.Magic = layersMagic
	meta.Embd = len(dump.Embed)
	meta.NumLayers = len(dump.Layers)
	meta.EmbedRMS = rmsOf(dump.Embed)
	meta.LayerRMS = make([]float32, len(dump.Layers))
	for i, ly := range dump.Layers {
		meta.LayerRMS[i] = rmsOf(ly)
	}

	f, err := os.Create(path + ".bin")
	if err != nil {
		return err
	}

	if err := WriteLayersBin(f, dump); err != nil {
		f.Close()
		return err
	}

	if err := f.Close(); err != nil {
		return err
	}

	js, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path+".json", append(js, '\n'), 0o644)
}

// LoadLayersDump читает path.bin (+ json)
func LoadLayersDump(path string) (LayersMeta, LayersDump, error) {
	var meta LayersMeta
	if data, err := os.ReadFile(path + ".json"); err == nil {
		_ = json.Unmarshal(data, &meta)
	}

	f, err := os.Open(path + ".bin")
	if err != nil {
		f, err = os.Open(path)
		if err != nil {
			return meta, LayersDump{}, err
		}
	}
	defer f.Close()

	dump, err := ReadLayersBin(f)

	return meta, dump, err
}

// DiffLayers сравнивает два dump; tol для OverTol
func DiffLayers(a, b LayersDump, tol float64) (embed DiffStats, layers []DiffStats, err error) {
	if len(a.Embed) != len(b.Embed) {
		return DiffStats{}, nil, fmt.Errorf("debug: embd A=%d B=%d", len(a.Embed), len(b.Embed))
	}

	if len(a.Layers) != len(b.Layers) {
		return DiffStats{}, nil, fmt.Errorf("debug: layers A=%d B=%d", len(a.Layers), len(b.Layers))
	}

	embed = DiffLogits(a.Embed, b.Embed, tol)
	layers = make([]DiffStats, len(a.Layers))
	for i := range a.Layers {
		layers[i] = DiffLogits(a.Layers[i], b.Layers[i], tol)
	}

	return embed, layers, nil
}

func rmsOf(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}

	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}

	return float32(math.Sqrt(sum / float64(len(x))))
}
