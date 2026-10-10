package hf

import (
	"fmt"
	"os"
	"strings"
)

// Resolve downloads (if needed) a GGUF from an HF repo and returns the local path.
// repoWithTag: "owner/repo" or "owner/repo:Q8_0".
func Resolve(repoWithTag string) (string, error) {
	spec, err := SplitRepoTag(repoWithTag)
	if err != nil {
		return "", err
	}

	token := Token()
	files, err := fetchRepoFilesOnline(spec.Repo, token)
	if err != nil {
		// offline fallback: local cache only
		cached := listCachedFiles(spec.Repo)
		if len(cached) == 0 {
			return "", fmt.Errorf("hf: failed to list files for %s: %w", spec.Repo, err)
		}

		fmt.Fprintf(os.Stderr, "hf: network unavailable, using cache (%v)\n", err)
		files = cached
	}

	primary, ok := FindBestModel(files, spec.Tag)
	if !ok {
		avail := ListModelGGUF(files)
		msg := fmt.Sprintf("hf: GGUF not found in %s", spec.Repo)
		if spec.Tag != "" {
			msg += fmt.Sprintf(" for tag %q", spec.Tag)
		}

		if len(avail) > 0 {
			msg += "\navailable files:\n  " + strings.Join(avail, "\n  ")
		}

		return "", fmt.Errorf("%s", msg)
	}

	// for offline cache URL/OID may be empty - file is already on disk
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

		return "", fmt.Errorf("hf: file %s is in cache index but missing on disk", primary.Path)
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
		return nil, fmt.Errorf("invalid repository id: %s", repoID)
	}

	commit, err := resolveCommit(repoID, token)
	if err != nil {
		return nil, err
	}

	return fetchRepoFiles(repoID, commit, token)
}
