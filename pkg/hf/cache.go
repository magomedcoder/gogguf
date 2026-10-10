package hf

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// CacheDir returns the HF Hub cache directory.
// Order: LLAMA_CACHE -> HF_HUB_CACHE -> HUGGINGFACE_HUB_CACHE -> HF_HOME/hub -> XDG_CACHE_HOME/huggingface/hub -> ~/.cache/huggingface/hub.
func CacheDir() (string, error) {
	type entry struct {
		env  string
		tail string // relative to env value; empty = use as-is
	}
	entries := []entry{
		{"LLAMA_CACHE", ""},
		{"HF_HUB_CACHE", ""},
		{"HUGGINGFACE_HUB_CACHE", ""},
		{"HF_HOME", "hub"},
		{"XDG_CACHE_HOME", filepath.Join("huggingface", "hub")},
	}

	for _, e := range entries {
		if v := os.Getenv(e.env); v != "" {
			if e.tail == "" {
				return v, nil
			}
			return filepath.Join(v, e.tail), nil
		}
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to resolve HF cache: %w", err)
	}

	return filepath.Join(home, ".cache", "huggingface", "hub"), nil
}

// RepoFolderName converts "owner/repo" to "models--owner--repo".
func RepoFolderName(repoID string) string {
	return "models--" + strings.ReplaceAll(repoID, "/", "--")
}

// RepoPath returns the repository directory path in cache.
func RepoPath(repoID string) (string, error) {
	base, err := CacheDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(base, RepoFolderName(repoID)), nil
}

func writeAtomic(path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		return err
	}

	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}

	return nil
}
