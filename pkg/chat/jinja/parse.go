package jinja

import "slices"

import "strings"

type parser struct {
	toks             []token
	pos              int
	pendingTrimRight bool
}

func parse(toks []token) (*program, error) {
	p := &parser{toks: toks}
	nodes, err := p.parseBlockUntil(nil)
	if err != nil {
		return nil, err
	}

	if !p.at(tokEOF) {
		return nil, errf("лишние токены после шаблона")
	}

	return &program{nodes: nodes}, nil
}

func (p *parser) parseBlockUntil(endKeywords []string) ([]node, error) {
	var nodes []node
	for !p.at(tokEOF) {
		if p.at(tokStmtOpen) {
			save := p.pos
			p.advance()
			if p.at(tokIdent) {
				kw := p.advance().text
				if slices.Contains(endKeywords, kw) {
					p.pos = save
					return nodes, nil
				}
			}
			p.pos = save

			trimL := p.advance().trimLeft
			if trimL && len(nodes) > 0 {
				p.trimLastText(&nodes)
			}

			if !p.at(tokIdent) {
				return nil, errf("ожидали ключевое слово блока")
			}

			kw := p.advance().text
			stmtNodes, err := p.parseStmt(kw)
			if err != nil {
				return nil, err
			}

			nodes = append(nodes, stmtNodes...)
			continue
		}

		if p.pendingTrimRight {
			p.pendingTrimRight = false
			if p.at(tokText) {
				t := p.advance().text
				nodes = append(nodes, textNode{text: stringsTrimLeft(t)})
				continue
			}
		}

		if p.at(tokText) {
			nodes = append(nodes, textNode{text: p.advance().text})
			continue
		}

		if p.at(tokExprOpen) {
			trimL := p.advance().trimLeft
			if trimL && len(nodes) > 0 {
				p.trimLastText(&nodes)
			}

			expr, err := p.parseExpr()
			if err != nil {
				return nil, err
			}

			if !p.at(tokExprClose) {
				return nil, errf("ожидали }}")
			}

			trimR := p.advance().trimRight
			nodes = append(nodes, exprNode{expr: expr})
			if trimR {
				p.pendingTrimRight = true
			}
			continue
		}

		return nil, errf("неожиданный токен %v", p.peek().typ)
	}
	return nodes, nil
}

func (p *parser) parseStmt(kw string) ([]node, error) {
	switch kw {
	case "if":
		return p.parseIf()
	case "for":
		return p.parseFor()
	case "set":
		return p.parseSet()
	default:
		return nil, errf("неизвестный блок %s", kw)
	}
}

func (p *parser) parseIf() ([]node, error) {
	cond, err := p.parseExpr()
	if err != nil {
		return nil, err
	}

	if trimR, err := p.expectStmtClose(); err != nil {
		return nil, err
	} else if trimR {
		p.pendingTrimRight = true
	}

	body, err := p.parseBlockUntil([]string{"elif", "else", "endif"})
	if err != nil {
		return nil, err
	}

	var elifs []elifBranch
	for {
		if !p.at(tokStmtOpen) {
			break
		}

		save := p.pos

		p.advance()

		if !p.at(tokIdent) || p.advance().text != "elif" {
			p.pos = save
			break
		}

		econd, err := p.parseExpr()
		if err != nil {

			return nil, err
		}
		if trimR, err := p.expectStmtClose(); err != nil {
			return nil, err
		} else if trimR {
			p.pendingTrimRight = true
		}

		eb, err := p.parseBlockUntil([]string{"elif", "else", "endif"})
		if err != nil {
			return nil, err
		}

		elifs = append(elifs, elifBranch{cond: econd, body: eb})
	}

	var elseBody []node
	if p.at(tokStmtOpen) {
		save := p.pos
		p.advance()
		if p.at(tokIdent) && p.advance().text == "else" {
			if trimR, err := p.expectStmtClose(); err != nil {
				return nil, err
			} else if trimR {
				p.pendingTrimRight = true
			}

			elseBody, err = p.parseBlockUntil([]string{"endif"})
			if err != nil {
				return nil, err
			}
		} else {
			p.pos = save
		}
	}

	if !p.matchStmtKeyword("endif") {
		return nil, errf("ожидали endif")
	}

	return []node{ifNode{
		cond:     cond,
		body:     body,
		elifs:    elifs,
		elseBody: elseBody,
	}}, nil
}

func (p *parser) parseFor() ([]node, error) {
	if !p.at(tokIdent) {
		return nil, errf("ожидали переменную цикла")
	}

	vars := []string{p.advance().text}
	if p.at(tokComma) {
		p.advance()
		if !p.at(tokIdent) {
			return nil, errf("ожидали переменную цикла")
		}
		vars = append(vars, p.advance().text)
	}

	if p.at(tokIn) {
		p.advance()
	} else if p.at(tokIdent) && p.peek().text == "in" {
		p.advance()
	} else {
		return nil, errf("ожидали in")
	}

	iter, err := p.parseExpr()
	if err != nil {
		return nil, err
	}

	if trimR, err := p.expectStmtClose(); err != nil {
		return nil, err
	} else if trimR {
		p.pendingTrimRight = true
	}

	body, err := p.parseBlockUntil([]string{"else", "endfor"})
	if err != nil {
		return nil, err
	}

	var elseBody []node
	if p.at(tokStmtOpen) {
		save := p.pos
		p.advance()
		if p.at(tokIdent) && p.advance().text == "else" {
			if trimR, err := p.expectStmtClose(); err != nil {
				return nil, err
			} else if trimR {
				p.pendingTrimRight = true
			}

			elseBody, err = p.parseBlockUntil([]string{"endfor"})
			if err != nil {
				return nil, err
			}
		} else {
			p.pos = save
		}
	}

	if !p.matchStmtKeyword("endfor") {
		return nil, errf("ожидали endfor")
	}

	return []node{forNode{
		vars:     vars,
		iter:     iter,
		body:     body,
		elseBody: elseBody,
	}}, nil
}

func (p *parser) parseSet() ([]node, error) {
	var assigns []assignment
	for {
		target, err := p.parseSetTarget()
		if err != nil {
			return nil, err
		}

		if !p.at(tokAssign) {
			return nil, errf("ожидали = в set")
		}

		p.advance()
		val, err := p.parseExpr()
		if err != nil {
			return nil, err
		}

		assigns = append(assigns, assignment{
			target: target,
			value:  val,
		})
		if !p.at(tokComma) {
			break
		}

		p.advance()
	}
	if trimR, err := p.expectStmtClose(); err != nil {
		return nil, err
	} else if trimR {
		p.pendingTrimRight = true
	}

	return []node{setNode{
		assigns: assigns,
	}}, nil
}

func (p *parser) expectStmtClose() (bool, error) {
	if !p.at(tokStmtClose) {
		return false, errf("ожидали %%}")
	}

	trimR := p.advance().trimRight

	return trimR, nil
}

func (p *parser) parseSetTarget() (expr, error) {
	e, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}

	for p.at(tokDot) {
		p.advance()
		if !p.at(tokIdent) {
			return nil, errf("ожидали имя атрибута")
		}
		e = attrExpr{
			obj:  e,
			attr: p.advance().text,
		}
	}

	return e, nil
}

func (p *parser) matchStmtKeyword(kw string) bool {
	if !p.at(tokStmtOpen) {
		return false
	}

	p.advance()
	if !p.at(tokIdent) || p.advance().text != kw {
		return false
	}

	if !p.at(tokStmtClose) {
		return false
	}

	p.advance()

	return true
}

func (p *parser) trimLastText(nodes *[]node) {
	if len(*nodes) == 0 {
		return
	}

	if tn, ok := (*nodes)[len(*nodes)-1].(textNode); ok {
		(*nodes)[len(*nodes)-1] = textNode{text: stringsTrimRight(tn.text)}
	}
}

func (p *parser) parseExpr() (expr, error) {
	return p.parseOr()
}

func (p *parser) parseOr() (expr, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}

	for p.at(tokOr) {
		p.advance()
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = binExpr{
			op:    tokOr,
			left:  left,
			right: right,
		}
	}

	return left, nil
}

func (p *parser) parseAnd() (expr, error) {
	left, err := p.parseNot()
	if err != nil {
		return nil, err
	}

	for p.at(tokAnd) {
		p.advance()
		right, err := p.parseNot()
		if err != nil {
			return nil, err
		}

		left = binExpr{
			op:    tokAnd,
			left:  left,
			right: right,
		}
	}

	return left, nil
}

func (p *parser) parseNot() (expr, error) {
	if p.at(tokNot) {
		p.advance()
		e, err := p.parseNot()
		if err != nil {
			return nil, err
		}

		return unaryExpr{
			op:   tokNot,
			expr: e,
		}, nil
	}

	return p.parseCompare()
}

func (p *parser) parseCompare() (expr, error) {
	left, err := p.parseAdd()
	if err != nil {
		return nil, err
	}

	if p.at(tokIs) {
		p.advance()
		not := false
		if p.at(tokNot) {
			p.advance()
			not = true
		}
		if !p.at(tokIdent) {
			return nil, errf("ожидали имя теста после is")
		}

		name := p.advance().text
		return testExpr{
			left: left,
			name: name,
			not:  not,
		}, nil
	}

	for p.at(tokIn) || p.at(tokEq) || p.at(tokNe) || p.at(tokLt) || p.at(tokGt) || p.at(tokLe) || p.at(tokGe) {
		op := p.advance().typ
		right, err := p.parseAdd()
		if err != nil {
			return nil, err
		}

		left = binExpr{
			op:    op,
			left:  left,
			right: right,
		}
	}

	return left, nil
}

func (p *parser) parseAdd() (expr, error) {
	left, err := p.parseMul()
	if err != nil {
		return nil, err
	}

	for p.at(tokPlus) || p.at(tokMinus) {
		op := p.advance().typ
		right, err := p.parseMul()
		if err != nil {
			return nil, err
		}

		left = binExpr{
			op:    op,
			left:  left,
			right: right,
		}
	}

	return left, nil
}

func (p *parser) parseMul() (expr, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}

	for p.at(tokStar) || p.at(tokSlash) || p.at(tokPercent) {
		op := p.advance().typ
		right, err := p.parseUnary()
		if err != nil {
			return nil, err
		}

		left = binExpr{
			op:    op,
			left:  left,
			right: right,
		}
	}

	return left, nil
}

func (p *parser) parseUnary() (expr, error) {
	if p.at(tokMinus) {
		p.advance()
		e, err := p.parseUnary()
		if err != nil {
			return nil, err
		}

		return unaryExpr{
			op:   tokMinus,
			expr: e,
		}, nil
	}

	return p.parsePostfix()
}

func (p *parser) parsePostfix() (expr, error) {
	e, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}

	for {
		switch {
		case p.at(tokDot):
			p.advance()
			if !p.at(tokIdent) {
				return nil, errf("ожидали имя атрибута")
			}

			e = attrExpr{
				obj:  e,
				attr: p.advance().text,
			}
		case p.at(tokLBracket):
			p.advance()
			key, err := p.parseExpr()
			if err != nil {
				return nil, err
			}

			if !p.at(tokRBracket) {
				return nil, errf("ожидали ]")
			}

			p.advance()
			e = indexExpr{
				obj: e,
				key: key,
			}
		case p.at(tokLParen):
			p.advance()
			args, err := p.parseArgs()
			if err != nil {
				return nil, err
			}

			e = callExpr{
				fn:   e,
				args: args,
			}
		case p.at(tokPipe):
			p.advance()
			if !p.at(tokIdent) {
				return nil, errf("ожидали имя фильтра")
			}
			name := p.advance().text
			var args []expr
			if p.at(tokLParen) {
				p.advance()
				args, err = p.parseArgs()
				if err != nil {
					return nil, err
				}
			}

			e = filterExpr{
				input: e,
				name:  name,
				args:  args,
			}
		default:
			return e, nil
		}
	}
}

func (p *parser) parsePrimary() (expr, error) {
	switch {
	case p.at(tokString):
		return litExpr{
			value: strVal(p.advance().text),
		}, nil
	case p.at(tokNumber):
		return litExpr{
			value: numVal(p.advance().text),
		}, nil
	case p.at(tokTrue):
		p.advance()
		return litExpr{
			value: boolVal(true),
		}, nil
	case p.at(tokFalse):
		p.advance()
		return litExpr{
			value: boolVal(false),
		}, nil
	case p.at(tokNone):
		p.advance()
		return litExpr{
			value: noneVal(),
		}, nil
	case p.at(tokIdent):
		return varExpr{
			name: p.advance().text,
		}, nil
	case p.at(tokLParen):
		p.advance()
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if !p.at(tokRParen) {
			return nil, errf("ожидали )")
		}
		p.advance()
		return e, nil
	case p.at(tokLBracket):
		p.advance()
		var items []expr
		if !p.at(tokRBracket) {
			for {
				item, err := p.parseExpr()
				if err != nil {
					return nil, err
				}

				items = append(items, item)
				if p.at(tokComma) {
					p.advance()
					if p.at(tokRBracket) {
						break
					}

					continue
				}

				break
			}
		}

		if !p.at(tokRBracket) {
			return nil, errf("ожидали ]")
		}

		p.advance()

		return arrayLiteralExpr{
			items: items,
		}, nil
	case p.at(tokLBrace):
		return p.parseObjectLiteral()
	default:
		return nil, errf("неожиданный токен в выражении")
	}
}

func (p *parser) parseObjectLiteral() (expr, error) {
	p.advance()
	obj := make(map[string]expr)
	if !p.at(tokRBrace) {
		for {
			var key string
			if p.at(tokString) {
				key = p.advance().text
			} else if p.at(tokIdent) {
				key = p.advance().text
			} else {
				return nil, errf("ожидали ключ объекта")
			}

			if !p.at(tokColon) {
				return nil, errf("ожидали :")
			}

			p.advance()
			valExpr, err := p.parseExpr()
			if err != nil {
				return nil, err
			}

			obj[key] = valExpr
			if p.at(tokComma) {
				p.advance()
				if p.at(tokRBrace) {
					break
				}
				continue
			}
			break
		}
	}

	if !p.at(tokRBrace) {
		return nil, errf("ожидали }")
	}

	p.advance()

	return objectLiteralExpr{
		fields: obj,
	}, nil
}

func (p *parser) parseArgs() ([]expr, error) {
	var args []expr
	if p.at(tokRParen) {
		p.advance()
		return args, nil
	}

	for {
		if p.at(tokIdent) && p.pos+1 < len(p.toks) && p.toks[p.pos+1].typ == tokAssign {
			name := p.advance().text
			p.advance()
			val, err := p.parseExpr()
			if err != nil {
				return nil, err
			}

			args = append(args, kwargExpr{
				name:  name,
				value: val,
			})

		} else {
			e, err := p.parseExpr()
			if err != nil {
				return nil, err
			}

			args = append(args, e)
		}

		if p.at(tokComma) {
			p.advance()
			continue
		}

		if !p.at(tokRParen) {
			return nil, errf("ожидали )")
		}

		p.advance()

		return args, nil
	}
}

func (p *parser) peek() token {
	if p.pos >= len(p.toks) {
		return token{typ: tokEOF}
	}

	return p.toks[p.pos]
}

func (p *parser) at(t tokenType) bool {
	return p.peek().typ == t
}

func (p *parser) advance() token {
	t := p.peek()
	p.pos++
	return t
}

func stringsTrimRight(s string) string {
	return strings.TrimRight(s, " \t\n\r")
}

func stringsTrimLeft(s string) string {
	return strings.TrimLeft(s, " \t\n\r")
}
