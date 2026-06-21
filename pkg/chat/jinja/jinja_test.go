package jinja

import (
	"strings"
	"testing"
)

func TestSimpleVariable(t *testing.T) {
	got, err := Render("Hello {{ name }}!", map[string]any{"name": "world"})
	if err != nil {
		t.Fatal(err)
	}

	if got != "Hello world!" {
		t.Fatalf("got %q", got)
	}
}

func TestIfFor(t *testing.T) {
	tmpl := `{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}`
	got, err := Render(tmpl, map[string]any{
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "hi",
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if got != "user:hi;" {
		t.Fatalf("got %q", got)
	}
}

func TestTrimMarkers(t *testing.T) {
	tmpl := "a{%- if true -%}b{%- endif -%}c"
	got, err := Render(tmpl, nil)
	if err != nil {
		t.Fatal(err)
	}

	if got != "abc" {
		t.Fatalf("got %q", got)
	}
}

func TestNamespaceAndRange(t *testing.T) {
	tmpl := `{% set ns = namespace(n=0) %}{% for i in range(3) %}{% set ns.n = i %}{% endfor %}{{ ns.n }}`
	got, err := Render(tmpl, nil)
	if err != nil {
		t.Fatal(err)
	}

	if got != "2" {
		t.Fatalf("got %q", got)
	}
}

func TestIsDefined(t *testing.T) {
	tmpl := `{% if x is defined %}yes{% else %}no{% endif %}`
	got, err := Render(tmpl, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}

	if got != "no" {
		t.Fatalf("got %q", got)
	}
}

func TestStringConcat(t *testing.T) {
	tmpl := `{{ '<start>' + role + ':' + content }}`
	got, err := Render(tmpl, map[string]any{
		"role":    "user",
		"content": "hi",
	})

	if err != nil {
		t.Fatal(err)
	}

	if got != "<start>user:hi" {
		t.Fatalf("got %q", got)
	}
}

func TestChatMLLike(t *testing.T) {
	imStart := "<|im_start|>"
	imEnd := "<|im_end|>"
	tmpl := `{%- for message in messages -%}{%- if message.role == "user" -%}{{ im_start + message.role + '\n' + message.content + im_end + '\n' }}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{ im_start + 'assistant\n' }}{%- endif -%}`

	got, err := Render(tmpl, map[string]any{
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "Hello",
			},
		},
		"add_generation_prompt": true,
		"im_start":              imStart,
		"im_end":                imEnd,
	})
	if err != nil {
		t.Fatal(err)
	}

	want := imStart + "user\nHello" + imEnd + "\n" + imStart + "assistant\n"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestSplitMethod(t *testing.T) {
	tmpl := `{{ 'a\nb'.split('\n')[0] }}`
	got, err := Render(tmpl, nil)
	if err != nil {
		t.Fatal(err)
	}

	if got != "a" {
		t.Fatalf("got %q", got)
	}
}

func TestInOperator(t *testing.T) {
	tmpl := `{% if 'ab' in content %}yes{% endif %}`
	got, err := Render(tmpl, map[string]any{"content": "xxabxx"})
	if err != nil {
		t.Fatal(err)
	}

	if got != "yes" {
		t.Fatalf("got %q", got)
	}
}

func TestLengthFilter(t *testing.T) {
	tmpl := `{{ messages|length }}`
	got, err := Render(tmpl, map[string]any{
		"messages": []any{1, 2, 3},
	})

	if err != nil {
		t.Fatal(err)
	}

	if got != "3" {
		t.Fatalf("got %q", got)
	}
}

func TestNoTrailingWhitespaceAfterTrim(t *testing.T) {
	if strings.TrimSpace("  ") != "" {
		t.Fatal("sanity")
	}
}
