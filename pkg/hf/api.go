package hf

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const defaultEndpoint = "https://huggingface.co/"

// Endpoint возвращает базовый URL Hub (MODEL_ENDPOINT или huggingface.co)
func Endpoint() string {
	ep := strings.TrimSpace(os.Getenv("MODEL_ENDPOINT"))
	if ep == "" {
		ep = defaultEndpoint
	}

	if !strings.HasSuffix(ep, "/") {
		ep += "/"
	}

	return ep
}

// Token возвращает HF_TOKEN из окружения.
func Token() string {
	return strings.TrimSpace(os.Getenv("HF_TOKEN"))
}

type httpClient struct {
	client *http.Client
	token  string
}

func newHTTPClient(token string) *httpClient {
	return &httpClient{
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
		token: token,
	}
}

func (c *httpClient) getJSON(url string, dest any) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}

	req.Header.Set("User-Agent", "gogguf/hf")
	req.Header.Set("Accept", "application/json")

	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		msg := string(body)
		var errObj struct {
			Error string `json:"error"`
		}

		if json.Unmarshal(body, &errObj) == nil && errObj.Error != "" {
			msg = errObj.Error
		}

		return fmt.Errorf("GET %s: %s (%s)", url, resp.Status, msg)
	}

	return json.Unmarshal(body, dest)
}

type refsResponse struct {
	Branches []struct {
		Name         string `json:"name"`
		TargetCommit string `json:"targetCommit"`
	} `json:"branches"`
}

type treeItem struct {
	Type string `json:"type"`
	Path string `json:"path"`
	OID  string `json:"oid"`
	Size int64  `json:"size"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}

// resolveCommit получает SHA ветки main (или первой) и пишет в refs/
func resolveCommit(repoID, token string) (string, error) {
	cli := newHTTPClient(token)
	url := Endpoint() + "api/models/" + repoID + "/refs"
	var refs refsResponse
	if err := cli.getJSON(url, &refs); err != nil {
		return "", err
	}

	var name, commit string
	for _, b := range refs.Branches {
		if !validCommit(b.TargetCommit) {
			continue
		}

		if b.Name == "main" {
			name, commit = b.Name, b.TargetCommit
			break
		}

		if name == "" {
			name, commit = b.Name, b.TargetCommit
		}
	}

	if commit == "" {
		return "", fmt.Errorf("нет валидной ветки для %s", repoID)
	}

	repoPath, err := RepoPath(repoID)
	if err != nil {
		return "", err
	}

	if err := writeAtomic(filepath.Join(repoPath, "refs", name), commit+"\n"); err != nil {
		return "", err
	}

	return commit, nil
}

// fetchRepoFiles запрашивает дерево файлов репозитория
func fetchRepoFiles(repoID, commit, token string) ([]FileInfo, error) {
	cli := newHTTPClient(token)
	url := Endpoint() + "api/models/" + repoID + "/tree/" + commit + "?recursive=true"
	var items []treeItem
	if err := cli.getJSON(url, &items); err != nil {
		return nil, err
	}

	repoPath, err := RepoPath(repoID)
	if err != nil {
		return nil, err
	}
	blobsPath := filepath.Join(repoPath, "blobs")
	commitPath := filepath.Join(repoPath, "snapshots", commit)
	ep := Endpoint()

	var files []FileInfo
	for _, item := range items {
		if item.Type != "file" || item.Path == "" {
			continue
		}

		if strings.Contains(item.Path, "..") || filepath.IsAbs(item.Path) {
			continue
		}

		f := FileInfo{
			RepoID: repoID,
			Path:   item.Path,
			URL:    ep + repoID + "/resolve/" + commit + "/" + item.Path,
		}

		if item.LFS != nil {
			f.OID = item.LFS.OID
			f.Size = item.LFS.Size
		} else {
			f.OID = item.OID
		}

		if f.Size == 0 {
			f.Size = item.Size
		}

		if f.OID != "" && !validOID(f.OID) {
			continue
		}

		f.FinalPath = filepath.Join(commitPath, filepath.FromSlash(item.Path))
		if f.OID != "" {
			if _, err := os.Stat(f.FinalPath); err == nil {
				f.LocalPath = f.FinalPath
			} else {
				f.LocalPath = filepath.Join(blobsPath, f.OID)
			}
		} else {
			f.LocalPath = f.FinalPath
		}

		files = append(files, f)
	}

	return files, nil
}

// cachedCommit читает refs/main (или любой ref) из локального кэша
func cachedCommit(repoID string) string {
	repoPath, err := RepoPath(repoID)
	if err != nil {
		return ""
	}

	refsPath := filepath.Join(repoPath, "refs")
	entries, err := os.ReadDir(refsPath)
	if err != nil {
		return ""
	}

	var fallback string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}

		data, err := os.ReadFile(filepath.Join(refsPath, e.Name()))
		if err != nil {
			continue
		}

		commit := strings.TrimSpace(string(data))
		if !validCommit(commit) {
			continue
		}

		if e.Name() == "main" {
			return commit
		}

		if fallback == "" {
			fallback = commit
		}
	}

	return fallback
}

// listCachedFiles перечисляет файлы в snapshots/{commit} для репо
func listCachedFiles(repoID string) []FileInfo {
	commit := cachedCommit(repoID)
	if commit == "" {
		return nil
	}

	repoPath, err := RepoPath(repoID)
	if err != nil {
		return nil
	}

	commitPath := filepath.Join(repoPath, "snapshots", commit)
	var files []FileInfo
	_ = filepath.Walk(commitPath, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(commitPath, p)
		if err != nil {
			return nil
		}

		files = append(files, FileInfo{
			RepoID:    repoID,
			Path:      filepath.ToSlash(rel),
			LocalPath: p,
			FinalPath: p,
		})

		return nil
	})

	return files
}
