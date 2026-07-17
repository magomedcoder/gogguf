package hf

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSplitRepoTag(t *testing.T) {
	tests := []struct {
		in      string
		repo    string
		tag     string
		wantErr bool
	}{
		{"Qwen/Qwen3-0.6B-GGUF:Q8_0", "Qwen/Qwen3-0.6B-GGUF", "Q8_0", false},
		{"bartowski/Llama-3.2-1B-Instruct-GGUF", "bartowski/Llama-3.2-1B-Instruct-GGUF", "", false},
		{"owner/repo:q4_k_m", "owner/repo", "q4_k_m", false},
		{"", "", "", true},
		{"noslash", "", "", true},
		{"a/b/c", "", "", true},
		{"/bad", "", "", true},
		{"bad/", "", "", true},
	}
	for _, tt := range tests {
		spec, err := SplitRepoTag(tt.in)
		if tt.wantErr {
			if err == nil {
				t.Fatalf("%q: ожидалась ошибка", tt.in)
			}
			continue
		}

		if err != nil {
			t.Fatalf("%q: %v", tt.in, err)
		}

		if spec.Repo != tt.repo || spec.Tag != tt.tag {
			t.Fatalf("%q: got %+v, want repo=%q tag=%q", tt.in, spec, tt.repo, tt.tag)
		}
	}
}

func TestValidRepoID(t *testing.T) {
	ok := []string{"Qwen/Qwen3-0.6B-GGUF", "a/b", "org_name/model-name.v1"}
	bad := []string{"", "a", "a/b/c", "/a/b", "a/b/", "a//b", "a /b"}
	for _, s := range ok {
		if !ValidRepoID(s) {
			t.Fatalf("ожидался валидный id: %q", s)
		}
	}

	for _, s := range bad {
		if ValidRepoID(s) {
			t.Fatalf("ожидался невалидный id: %q", s)
		}
	}
}

func TestFindBestModel(t *testing.T) {
	files := []FileInfo{
		{
			Path: "mmproj-f16.gguf",
		},
		{
			Path: "model-Q4_0.gguf",
		},
		{
			Path: "model-Q4_K_M.gguf",
		},
		{
			Path: "model-Q8_0.gguf",
		},
		{
			Path: "imatrix.gguf",
		},
	}

	got, ok := FindBestModel(files, "Q8_0")
	if !ok || got.Path != "model-Q8_0.gguf" {
		t.Fatalf("tag Q8_0: got %+v ok=%v", got, ok)
	}

	got, ok = FindBestModel(files, "q4_k_m")
	if !ok || got.Path != "model-Q4_K_M.gguf" {
		t.Fatalf("tag q4_k_m: got %+v ok=%v", got, ok)
	}

	got, ok = FindBestModel(files, "")
	if !ok || got.Path != "model-Q4_K_M.gguf" {
		t.Fatalf("default: got %+v ok=%v", got, ok)
	}

	got, ok = FindBestModel(files, "Q5_K_M")
	if ok {
		t.Fatalf("отсутствующий тег не должен матчиться: %+v", got)
	}

	onlyQ8 := []FileInfo{
		{
			Path: "only-Q8_0.gguf",
		},
	}
	got, ok = FindBestModel(onlyQ8, "")
	if !ok || got.Path != "only-Q8_0.gguf" {
		t.Fatalf("fallback Q8_0: got %+v ok=%v", got, ok)
	}
}

func TestIsModelGGUF(t *testing.T) {
	if !IsModelGGUF("foo/bar-Q8_0.gguf") {
		t.Fatal("ожидался model gguf")
	}

	if IsModelGGUF("mmproj-Q8_0.gguf") {
		t.Fatal("mmproj не должен считаться моделью")
	}

	if IsModelGGUF("readme.md") {
		t.Fatal("не-gguf")
	}
}

func TestRepoFolderName(t *testing.T) {
	got := RepoFolderName("Qwen/Qwen3-0.6B-GGUF")
	want := "models--Qwen--Qwen3-0.6B-GGUF"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestCacheDirEnv(t *testing.T) {
	t.Setenv("LLAMA_CACHE", "")
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("HUGGINGFACE_HUB_CACHE", "")
	t.Setenv("HF_HOME", "")
	t.Setenv("XDG_CACHE_HOME", "")

	tmp := t.TempDir()
	t.Setenv("HF_HUB_CACHE", tmp)
	dir, err := CacheDir()
	if err != nil {
		t.Fatal(err)
	}

	if dir != tmp {
		t.Fatalf("HF_HUB_CACHE: got %q want %q", dir, tmp)
	}

	t.Setenv("HF_HUB_CACHE", "")
	hfHome := filepath.Join(tmp, "hfhome")
	t.Setenv("HF_HOME", hfHome)
	dir, err = CacheDir()
	if err != nil {
		t.Fatal(err)
	}

	want := filepath.Join(hfHome, "hub")
	if dir != want {
		t.Fatalf("HF_HOME: got %q want %q", dir, want)
	}
}

func TestRepoPathUsesCache(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("LLAMA_CACHE", tmp)
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("HUGGINGFACE_HUB_CACHE", "")
	t.Setenv("HF_HOME", "")
	t.Setenv("XDG_CACHE_HOME", "")

	p, err := RepoPath("owner/model")
	if err != nil {
		t.Fatal(err)
	}

	want := filepath.Join(tmp, "models--owner--model")
	if p != want {
		t.Fatalf("got %q want %q", p, want)
	}

	if err := os.MkdirAll(p, 0o755); err != nil {
		t.Fatal(err)
	}
}
