package jinja

import (
	"fmt"
	"strings"
)

// Render выполняет Jinja2-шаблон с заданным контекстом
func Render(template string, ctx map[string]any) (string, error) {
	tokens, err := tokenize(template)
	if err != nil {
		return "", err
	}

	prog, err := parse(tokens)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	if err := execProgram(prog, ctx, &b); err != nil {
		return "", err
	}

	return b.String(), nil
}

func errf(format string, args ...any) error {
	return fmt.Errorf("jinja: "+format, args...)
}
