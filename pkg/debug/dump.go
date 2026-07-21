package debug

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

const logitsMagic = "GGUFLGT1"

// LogitsMeta - JSON-метаданные рядом с .bin dump
type LogitsMeta struct {
	Magic   string  `json:"magic"`
	Model   string  `json:"model"`
	Prompt  string  `json:"prompt"`
	Chat    bool    `json:"chat,omitempty"`
	Tokens  []int   `json:"tokens"`
	Vocab   int     `json:"vocab"`
	Greedy  int     `json:"greedy"`
	Top     []Logit `json:"top,omitempty"`
	Backend string  `json:"backend,omitempty"` // cpu / cuda
	NGL     int     `json:"ngl,omitempty"`
}

// WriteLogitsBin пишет float32 logits little-endian: magic(8) + n(u32) + data
func WriteLogitsBin(w io.Writer, logits []float32) error {
	if _, err := io.WriteString(w, logitsMagic); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint32(len(logits))); err != nil {
		return err
	}

	return binary.Write(w, binary.LittleEndian, logits)
}

// ReadLogitsBin читает dump от WriteLogitsBin
func ReadLogitsBin(r io.Reader) ([]float32, error) {
	magic := make([]byte, 8)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}

	if string(magic) != logitsMagic {
		return nil, fmt.Errorf("debug: неверный magic %q, ожидали %q", magic, logitsMagic)
	}

	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}

	logits := make([]float32, n)
	if err := binary.Read(r, binary.LittleEndian, logits); err != nil {
		return nil, err
	}

	return logits, nil
}

// SaveLogitsDump пишет path.bin и path.json
func SaveLogitsDump(path string, meta LogitsMeta, logits []float32) error {
	meta.Magic = logitsMagic
	meta.Vocab = len(logits)
	if meta.Greedy == 0 && len(logits) > 0 {
		meta.Greedy = greedyID(logits)
	}

	binPath := path + ".bin"
	f, err := os.Create(binPath)
	if err != nil {
		return err
	}

	if err := WriteLogitsBin(f, logits); err != nil {
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

// LoadLogitsDump читает path.bin (+ опционально path.json)
func LoadLogitsDump(path string) (LogitsMeta, []float32, error) {
	var meta LogitsMeta
	if data, err := os.ReadFile(path + ".json"); err == nil {
		_ = json.Unmarshal(data, &meta)
	}

	f, err := os.Open(path + ".bin")
	if err != nil {
		// допускаем path уже с .bin
		f, err = os.Open(path)
		if err != nil {
			return meta, nil, err
		}
	}
	defer f.Close()

	logits, err := ReadLogitsBin(f)
	return meta, logits, err
}

func greedyID(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}

	return best
}
