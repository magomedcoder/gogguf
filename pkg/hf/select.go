package hf

import (
	"path"
	"regexp"
	"strings"
)

// FileInfo is a file in an HF repository.
type FileInfo struct {
	RepoID    string
	Path      string // path inside repo, e.g. "model-Q8_0.gguf"
	OID       string
	Size      int64
	URL       string
	LocalPath string // blobs/{oid} or snapshots/... if already finalized
	FinalPath string // snapshots/{commit}/{path}
}

// IsModelGGUF is the main model GGUF (not mmproj/imatrix/etc.).
func IsModelGGUF(filepath string) bool {
	if !strings.HasSuffix(strings.ToLower(filepath), ".gguf") {
		return false
	}

	name := path.Base(filepath)
	lower := strings.ToLower(name)
	for _, skip := range []string{"mmproj", "imatrix", "mtp-", "eagle3-", "dflash-"} {
		if strings.Contains(lower, skip) {
			return false
		}
	}

	return true
}

// FindBestModel picks a GGUF by quantization tag.
// Without a tag: Q4_K_M, then Q8_0, otherwise the first suitable .gguf.
func FindBestModel(files []FileInfo, tag string) (FileInfo, bool) {
	var tags []string
	if tag != "" {
		tags = []string{tag}
	} else {
		tags = []string{"Q4_K_M", "Q8_0"}
	}

	for _, t := range tags {
		re, err := regexp.Compile("(?i)" + regexp.QuoteMeta(t) + `[.-]`)
		if err != nil {
			continue
		}

		for _, f := range files {
			if IsModelGGUF(f.Path) && re.FindStringIndex(f.Path) != nil {
				return f, true
			}
		}
	}

	if tag == "" {
		for _, f := range files {
			if IsModelGGUF(f.Path) {
				return f, true
			}
		}
	}

	return FileInfo{}, false
}

// ListModelGGUF returns paths of all main GGUF files in the list.
func ListModelGGUF(files []FileInfo) []string {
	var out []string
	for _, f := range files {
		if IsModelGGUF(f.Path) {
			out = append(out, f.Path)
		}
	}

	return out
}
