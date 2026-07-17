package hf

import (
	"fmt"
	"strings"
)

// RepoSpec - репозиторий Hugging Face с опциональным тегом квантизации
type RepoSpec struct {
	Repo string // owner/name
	Tag  string // например Q8_0; пусто = автовыбор
}

// SplitRepoTag разбирает "owner/repo[:quant]"
func SplitRepoTag(hfRepoWithTag string) (RepoSpec, error) {
	s := strings.TrimSpace(hfRepoWithTag)
	if s == "" {
		return RepoSpec{}, fmt.Errorf("пустой HF repo")
	}

	repo, tag := s, ""
	if i := strings.LastIndex(s, ":"); i >= 0 {
		repo = s[:i]
		tag = s[i+1:]
	}

	if !ValidRepoID(repo) {
		return RepoSpec{}, fmt.Errorf("неверный формат HF repo, ожидается owner/repo[:quant]: %q", hfRepoWithTag)
	}

	return RepoSpec{
		Repo: repo,
		Tag:  tag,
	}, nil
}

// ValidRepoID проверяет формат owner/repo (ровно один '/', допустимые символы)
func ValidRepoID(repoID string) bool {
	if repoID == "" || len(repoID) > 256 {
		return false
	}

	slash := 0
	special := true // предыдущий символ - «специальный» или начало
	for _, c := range repoID {
		switch {
		case isAlphanum(c) || c == '_':
			special = false
		case c == '/' || c == '.' || c == '-':
			if special {
				return false
			}

			if c == '/' {
				slash++
			}

			special = true
		default:
			return false
		}
	}

	return !special && slash == 1
}

func isAlphanum(c rune) bool {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
}

func isHexString(s string, n int) bool {
	if len(s) != n {
		return false
	}

	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}

	return true
}

func validCommit(hash string) bool {
	return isHexString(hash, 40)
}

func validOID(oid string) bool {
	return isHexString(oid, 40) || isHexString(oid, 64)
}
