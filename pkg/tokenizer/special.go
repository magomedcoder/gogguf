package tokenizer

import "sort"

type textSegment struct {
	text      string
	isSpecial bool
}

func (t *Tokenizer) buildSpecialTokens() {
	if t.special != nil {
		return
	}

	var special []string
	for _, tok := range t.tokens {
		if len(tok) >= 2 && tok[0] == '<' && t.id[tok] >= 0 {
			special = append(special, tok)
		}
	}

	if len(t.merges) == 0 {
		for _, tok := range []string{"[INST]", "[/INST]"} {
			if _, ok := t.id[tok]; ok {
				special = append(special, tok)
			}
		}
	}

	sort.Slice(special, func(i, j int) bool {
		return len(special[i]) > len(special[j])
	})

	t.special = special
}

func splitBySpecial(text string, special []string) []textSegment {
	if len(special) == 0 {
		return []textSegment{{text: text}}
	}

	var out []textSegment
	for len(text) > 0 {
		bestIdx := -1
		bestTok := ""
		for _, tok := range special {
			if len(text) >= len(tok) && text[:len(tok)] == tok {
				bestIdx = 0
				bestTok = tok
				break
			}
		}

		if bestIdx == 0 {
			out = append(out, textSegment{text: bestTok, isSpecial: true})
			text = text[len(bestTok):]
			continue
		}

		next := len(text)
		for _, tok := range special {
			if idx := indexOf(text, tok); idx >= 0 && idx < next {
				next = idx
			}
		}

		if next > 0 {
			out = append(out, textSegment{text: text[:next]})
			text = text[next:]
			continue
		}

		out = append(out, textSegment{text: text})
		break
	}

	return out
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}

	return -1
}
