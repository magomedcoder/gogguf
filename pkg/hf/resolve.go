package hf

import (
	"fmt"
	"os"
	"strings"
)

// Resolve скачивает (при необходимости) GGUF из HF repo и возвращает локальный путь.
// repoWithTag: "owner/repo" или "owner/repo:Q8_0".
func Resolve(repoWithTag string) (string, error) {
	spec, err := SplitRepoTag(repoWithTag)
	if err != nil {
		return "", err
	}

	token := Token()
	files, err := fetchRepoFilesOnline(spec.Repo, token)
	if err != nil {
		// offline fallback: только локальный кэш
		cached := listCachedFiles(spec.Repo)
		if len(cached) == 0 {
			return "", fmt.Errorf("hf: не удалось получить список файлов для %s: %w", spec.Repo, err)
		}

		fmt.Fprintf(os.Stderr, "hf: сеть недоступна, используем кэш (%v)\n", err)
		files = cached
	}

	primary, ok := FindBestModel(files, spec.Tag)
	if !ok {
		avail := ListModelGGUF(files)
		msg := fmt.Sprintf("hf: GGUF не найден в %s", spec.Repo)
		if spec.Tag != "" {
			msg += fmt.Sprintf(" для тега %q", spec.Tag)
		}

		if len(avail) > 0 {
			msg += "\nдоступные файлы:\n  " + strings.Join(avail, "\n  ")
		}

		return "", fmt.Errorf("%s", msg)
	}

	// для offline-кэша URL/OID могут быть пустыми - файл уже на диске
	if primary.URL == "" {
		if primary.FinalPath != "" {
			if _, err := os.Stat(primary.FinalPath); err == nil {
				fmt.Fprintf(os.Stderr, "hf: %s -> %s\n", spec.Repo, primary.FinalPath)
				return primary.FinalPath, nil
			}
		}

		if primary.LocalPath != "" {
			if _, err := os.Stat(primary.LocalPath); err == nil {
				fmt.Fprintf(os.Stderr, "hf: %s -> %s\n", spec.Repo, primary.LocalPath)
				return primary.LocalPath, nil
			}
		}

		return "", fmt.Errorf("hf: файл %s есть в индексе кэша, но отсутствует на диске", primary.Path)
	}

	path, err := ensureDownloaded(primary, token)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "hf: %s -> %s\n", primary.Path, path)
	return path, nil
}

func fetchRepoFilesOnline(repoID, token string) ([]FileInfo, error) {
	if !ValidRepoID(repoID) {
		return nil, fmt.Errorf("неверный repository id: %s", repoID)
	}

	commit, err := resolveCommit(repoID, token)
	if err != nil {
		return nil, err
	}

	return fetchRepoFiles(repoID, commit, token)
}
