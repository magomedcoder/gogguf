package tokenizer

import (
	"regexp"
	"unicode"
)

// llama-bpe (Llama 3): числа до 3 цифр, регистрозависимые сокращения
// Go regexp не поддерживает (?!) - последний сегмент упрощён до \s+
var pretokenizeLlamaBPEPattern = regexp.MustCompile(`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)

const outOfRange = -1

type cptFlags struct {
	letter     bool
	number     bool
	whitespace bool
}

func flagsFor(r rune) cptFlags {
	return cptFlags{
		letter:     unicode.IsLetter(r),
		number:     unicode.IsNumber(r),
		whitespace: unicode.IsSpace(r),
	}
}

func pretokenizeQwen2(text string) []string {
	cpts := []rune(text)
	if len(cpts) == 0 {
		return nil
	}

	var out []string
	start := 0
	offsets := []int{len(cpts)}

	for _, span := range offsets {
		offsetIni := start
		offsetEnd := start + span
		start = offsetEnd

		prevEnd := offsetIni
		addToken := func(end int) {
			if end > prevEnd {
				out = append(out, string(cpts[prevEnd:end]))
			}
			prevEnd = end
		}

		get := func(pos int) rune {
			if offsetIni <= pos && pos < offsetEnd {
				return cpts[pos]
			}
			return outOfRange
		}

		getFlags := func(pos int) cptFlags {
			if offsetIni <= pos && pos < offsetEnd {
				return flagsFor(cpts[pos])
			}
			return cptFlags{}
		}

		for pos := offsetIni; pos < offsetEnd; {
			cpt := get(pos)
			flags := getFlags(pos)

			if cpt == '\'' && pos+1 < offsetEnd {
				next := unicode.ToLower(get(pos + 1))
				switch next {
				case 's', 't', 'm', 'd':
					pos += 2
					addToken(pos)
					continue
				}

				if pos+2 < offsetEnd {
					next2 := unicode.ToLower(get(pos + 2))
					if (next == 'r' && next2 == 'e') || (next == 'v' && next2 == 'e') || (next == 'l' && next2 == 'l') {
						pos += 3
						addToken(pos)
						continue
					}
				}
			}

			if cpt != '\r' && cpt != '\n' && !flags.number {
				if flags.letter || getFlags(pos+1).letter {
					pos++
					for getFlags(pos).letter {
						pos++
					}
					addToken(pos)
					continue
				}
			}

			if flags.number {
				pos++
				addToken(pos)
				continue
			}

			flags2 := flags
			if cpt == ' ' {
				flags2 = getFlags(pos + 1)
			}

			if !flags2.whitespace && !flags2.letter && !flags2.number && cpt != outOfRange && cpt != 0 {
				if cpt == ' ' {
					pos++
				}

				for {
					f := getFlags(pos)
					if f.whitespace || f.letter || f.number {
						break
					}

					if get(pos) == outOfRange {
						break
					}
					pos++
				}

				for {
					c := get(pos)
					if c != '\r' && c != '\n' {
						break
					}
					pos++
				}

				addToken(pos)
				continue
			}

			numWS := 0
			lastRN := 0
			for getFlags(pos + numWS).whitespace {
				c := get(pos + numWS)
				if c == '\r' || c == '\n' {
					lastRN = pos + numWS + 1
				}
				numWS++
			}

			if lastRN > 0 {
				pos = lastRN
				addToken(pos)
				continue
			}

			if numWS > 1 && get(pos+numWS) != outOfRange {
				pos += numWS - 1
				addToken(pos)
				continue
			}

			if numWS > 0 {
				pos += numWS
				addToken(pos)
				continue
			}

			pos++
			addToken(pos)
		}
	}

	return out
}

// pretokenizeGPT2 - упрощённый GPT-2 pretokenizer (RE2-safe)
func pretokenizeGPT2(text string) []string {
	return pretokenizePattern.FindAllString(text, -1)
}

func pretokenizeLlamaBPE(text string) []string {
	return pretokenizeLlamaBPEPattern.FindAllString(text, -1)
}
