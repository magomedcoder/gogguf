package main

import "testing"

func TestResolveModelPath(t *testing.T) {
	p, err := resolveModelPath("/tmp/m.gguf", "")
	if err != nil || p != "/tmp/m.gguf" {
		t.Fatalf("local: path=%q err=%v", p, err)
	}

	_, err = resolveModelPath("/tmp/m.gguf", "owner/repo")
	if err == nil {
		t.Fatal("expected error when -m and -hf together")
	}

	_, err = resolveModelPath("", "")
	if err == nil {
		t.Fatal("expected error without model source")
	}
}
