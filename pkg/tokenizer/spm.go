package tokenizer

import "fmt"

// encodeGreedyVocab кодирует текст жадным longest-match по vocab (Mistral / SentencePiece без merges)
func (t *Tokenizer) encodeGreedyVocab(text string) ([]int, error) {
	if text == "" {
		return nil, nil
	}

	var out []int
	i := 0
	wordStart := true
	for i < len(text) {
		if text[i] == ' ' {
			i++
			wordStart = true
			continue
		}

		bestID := -1
		bestLen := 0

		maxLen := min(len(text)-i, 48)

		for l := maxLen; l >= 1; l-- {
			if text[i+l-1] == ' ' {
				continue
			}

			cand := text[i : i+l]
			if id, ok := t.id[cand]; ok {
				bestID = id
				bestLen = l
				break
			}

			if wordStart {
				if id, ok := t.id["▁"+cand]; ok {
					bestID = id
					bestLen = l
					break
				}
			}
		}

		if bestID < 0 {
			return nil, fmt.Errorf("tokenizer: неизвестный токен %q", text[i:i+1])
		}

		out = append(out, bestID)
		i += bestLen
		wordStart = false
	}

	return out, nil
}
