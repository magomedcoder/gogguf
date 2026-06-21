package jinja

import (
	"strings"
	"unicode"
)

type tokenType int

const (
	tokText tokenType = iota
	tokExprOpen
	tokExprClose
	tokStmtOpen
	tokStmtClose
	tokIdent
	tokString
	tokNumber
	tokTrue
	tokFalse
	tokNone
	tokLParen
	tokRParen
	tokLBracket
	tokRBracket
	tokLBrace
	tokRBrace
	tokDot
	tokComma
	tokPipe
	tokColon
	tokEq
	tokNe
	tokLt
	tokGt
	tokLe
	tokGe
	tokPlus
	tokMinus
	tokStar
	tokSlash
	tokPercent
	tokAnd
	tokOr
	tokNot
	tokIn
	tokIs
	tokAssign
	tokEOF
)

type token struct {
	typ       tokenType
	text      string
	trimLeft  bool
	trimRight bool
}

func tokenize(src string) ([]token, error) {
	var out []token
	i := 0

	for i < len(src) {
		if i+1 < len(src) && src[i] == '{' {
			switch src[i+1] {
			case '{':
				trimLeft := i+2 < len(src) && src[i+2] == '-'
				start := i + 2
				if trimLeft {
					start++
				}

				after, trimRight, innerEnd, err := findClose(src, start, "}}")
				if err != nil {
					return nil, err
				}

				inner := src[start:innerEnd]
				exprToks, err := lexWords(inner, false)
				if err != nil {
					return nil, err
				}

				out = append(out, token{
					typ:      tokExprOpen,
					trimLeft: trimLeft,
				})
				out = append(out, exprToks...)
				out = append(out, token{
					typ:       tokExprClose,
					trimRight: trimRight,
				})
				i = after
				continue

			case '%':
				trimLeft := i+2 < len(src) && src[i+2] == '-'
				start := i + 2
				if trimLeft {
					start++
				}

				after, trimRight, innerEnd, err := findClose(src, start, "%}")
				if err != nil {
					return nil, err
				}

				inner := strings.TrimSpace(src[start:innerEnd])
				stmtToks, err := lexWords(inner, true)
				if err != nil {
					return nil, err
				}

				out = append(out, token{
					typ:      tokStmtOpen,
					trimLeft: trimLeft,
				})
				out = append(out, stmtToks...)
				out = append(out, token{
					typ:       tokStmtClose,
					trimRight: trimRight,
				})
				i = after
				continue
			}
		}

		start := i
		for i < len(src) {
			if i+1 < len(src) && src[i] == '{' && (src[i+1] == '{' || src[i+1] == '%') {
				break
			}
			i++
		}

		if i > start {
			out = append(out, token{
				typ:  tokText,
				text: src[start:i],
			})
		}
	}

	out = append(out, token{typ: tokEOF})
	return out, nil
}

func findClose(src string, start int, end string) (after int, trimRight bool, innerEnd int, err error) {
	for i := start; i < len(src); i++ {
		if strings.HasPrefix(src[i:], end) {
			trimRight = i > start && src[i-1] == '-'
			innerEnd = i
			if trimRight {
				innerEnd = i - 1
			}

			return i + len(end), trimRight, innerEnd, nil
		}
	}

	return 0, false, 0, errf("незакрытый тег")
}

func lexWords(src string, stmt bool) ([]token, error) {
	var out []token
	i := 0

	for {
		for i < len(src) && (src[i] == ' ' || src[i] == '\t' || src[i] == '\r' || src[i] == '\n') {
			i++
		}

		if i >= len(src) {
			break
		}

		switch src[i] {
		case '(':
			out = append(out, token{typ: tokLParen})
			i++
		case ')':
			out = append(out, token{typ: tokRParen})
			i++
		case '[':
			out = append(out, token{typ: tokLBracket})
			i++
		case ']':
			out = append(out, token{typ: tokRBracket})
			i++
		case '{':
			out = append(out, token{typ: tokLBrace})
			i++
		case '}':
			out = append(out, token{typ: tokRBrace})
			i++
		case '.':
			out = append(out, token{typ: tokDot})
			i++
		case ',':
			out = append(out, token{typ: tokComma})
			i++
		case '|':
			out = append(out, token{typ: tokPipe})
			i++
		case ':':
			out = append(out, token{typ: tokColon})
			i++
		case '+':
			out = append(out, token{typ: tokPlus})
			i++
		case '-':
			if i+1 < len(src) && src[i+1] >= '0' && src[i+1] <= '9' {
				j := i + 1
				for j < len(src) && (unicode.IsDigit(rune(src[j])) || src[j] == '.') {
					j++
				}

				out = append(out, token{typ: tokNumber, text: src[i:j]})
				i = j
			} else {
				out = append(out, token{typ: tokMinus})
				i++
			}
		case '*':
			out = append(out, token{typ: tokStar})
			i++
		case '/':
			out = append(out, token{typ: tokSlash})
			i++
		case '%':
			out = append(out, token{typ: tokPercent})
			i++
		case '=':
			if i+1 < len(src) && src[i+1] == '=' {
				out = append(out, token{typ: tokEq})
				i += 2
			} else if stmt {
				out = append(out, token{typ: tokAssign})
				i++
			} else {
				return nil, errf("неожиданный '='")
			}
		case '!':
			if i+1 < len(src) && src[i+1] == '=' {
				out = append(out, token{typ: tokNe})
				i += 2
			} else {
				return nil, errf("неожиданный '!'")
			}
		case '<':
			if i+1 < len(src) && src[i+1] == '=' {
				out = append(out, token{typ: tokLe})
				i += 2
			} else {
				out = append(out, token{typ: tokLt})
				i++
			}
		case '>':
			if i+1 < len(src) && src[i+1] == '=' {
				out = append(out, token{typ: tokGe})
				i += 2
			} else {
				out = append(out, token{typ: tokGt})
				i++
			}
		case '\'', '"':
			q := src[i]
			i++
			var sb strings.Builder
			for i < len(src) {
				if src[i] == '\\' && i+1 < len(src) {
					switch src[i+1] {
					case 'n':
						sb.WriteByte('\n')
					case 't':
						sb.WriteByte('\t')
					case 'r':
						sb.WriteByte('\r')
					case '\\', '\'', '"':
						sb.WriteByte(src[i+1])
					default:
						sb.WriteByte(src[i])
						sb.WriteByte(src[i+1])
					}
					i += 2
					continue
				}

				if src[i] == q {
					out = append(out, token{typ: tokString, text: sb.String()})
					i++
					goto next
				}

				sb.WriteByte(src[i])
				i++
			}
			return nil, errf("незакрытая строка")
		default:
			if src[i] >= '0' && src[i] <= '9' {
				j := i
				for j < len(src) && (unicode.IsDigit(rune(src[j])) || src[j] == '.') {
					j++
				}

				out = append(out, token{
					typ:  tokNumber,
					text: src[i:j],
				})

				i = j
			} else if isIdentStart(src[i]) {
				j := i

				for j < len(src) && isIdentPart(src[j]) {
					j++
				}

				word := src[i:j]
				i = j
				switch word {
				case "true":
					out = append(out, token{
						typ:  tokTrue,
						text: word,
					})
				case "false":
					out = append(out, token{
						typ:  tokFalse,
						text: word,
					})
				case "none":
					out = append(out, token{
						typ:  tokNone,
						text: word,
					})
				case "and":
					out = append(out, token{
						typ:  tokAnd,
						text: word,
					})
				case "or":
					out = append(out, token{
						typ:  tokOr,
						text: word,
					})
				case "not":
					out = append(out, token{
						typ:  tokNot,
						text: word,
					})
				case "in":
					out = append(out, token{
						typ:  tokIn,
						text: word,
					})
				case "is":
					out = append(out, token{
						typ:  tokIs,
						text: word,
					})
				default:
					out = append(out, token{
						typ:  tokIdent,
						text: word,
					})
				}
			} else {
				return nil, errf("неожиданный символ %q", string(src[i]))
			}
		}
	next:
	}

	return out, nil
}

func isIdentStart(b byte) bool {
	return unicode.IsLetter(rune(b)) || b == '_'
}

func isIdentPart(b byte) bool {
	return unicode.IsLetter(rune(b)) || unicode.IsDigit(rune(b)) || b == '_'
}
