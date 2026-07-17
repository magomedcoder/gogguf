package hf

import (
	"path"
	"regexp"
	"strings"
)

// FileInfo - файл в репозитории HF
type FileInfo struct {
	RepoID    string
	Path      string // путь внутри репо, например "model-Q8_0.gguf"
	OID       string
	Size      int64
	URL       string
	LocalPath string // blobs/{oid} или snapshots/... если уже финализирован
	FinalPath string // snapshots/{commit}/{path}
}

// IsModelGGUF - основной GGUF модели (не mmproj/imatrix/и т.п.)
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

// FindBestModel выбирает GGUF по тегу квантизации.
// Без тега: Q4_K_M, затем Q8_0, иначе первый подходящий .gguf.
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

// ListModelGGUF возвращает пути всех основных GGUF в списке
func ListModelGGUF(files []FileInfo) []string {
	var out []string
	for _, f := range files {
		if IsModelGGUF(f.Path) {
			out = append(out, f.Path)
		}
	}

	return out
}
