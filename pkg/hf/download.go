package hf

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

var (
	symlinkMu        sync.Mutex
	symlinksDisabled bool
)

// downloadFile скачивает URL в localPath с resume и прогрессом в stderr
func downloadFile(url, localPath, token string, expectedSize int64) error {
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return err
	}

	tmp := localPath + ".tmp"
	var offset int64
	if st, err := os.Stat(tmp); err == nil {
		offset = st.Size()
	}

	if expectedSize > 0 && offset >= expectedSize {
		return os.Rename(tmp, localPath)
	}

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "gogguf/hf")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}

	client := &http.Client{
		Timeout: 0,
	} // большие модели; прогресс через body

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		offset = 0 // сервер отдал файл целиком
	case http.StatusPartialContent:
		// resume OK
	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("скачивание %s: %s (%s)", url, resp.Status, string(body))
	}

	var f *os.File
	if offset > 0 && resp.StatusCode == http.StatusPartialContent {
		f, err = os.OpenFile(tmp, os.O_WRONLY|os.O_APPEND, 0o644)
	} else {
		offset = 0
		f, err = os.Create(tmp)
	}

	if err != nil {
		return err
	}
	defer f.Close()

	total := expectedSize
	if total <= 0 {
		if cl := resp.ContentLength; cl > 0 {
			total = offset + cl
		}
	}

	name := filepath.Base(localPath)
	if len(name) > 40 {
		name = name[:39] + "..."
	}

	progress := &progressWriter{
		w:          f,
		name:       name,
		downloaded: offset,
		total:      total,
	}
	if _, err := io.Copy(progress, resp.Body); err != nil {
		progress.finish(false)
		return err
	}
	progress.finish(true)

	if err := f.Close(); err != nil {
		return err
	}

	return os.Rename(tmp, localPath)
}

type progressWriter struct {
	w          io.Writer
	name       string
	downloaded int64
	total      int64
	lastPrint  time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.downloaded += int64(n)
	now := time.Now()
	if now.Sub(p.lastPrint) >= 200*time.Millisecond || err != nil {
		p.print()
		p.lastPrint = now
	}

	return n, err
}

func (p *progressWriter) print() {
	if p.total > 0 {
		pct := 100 * p.downloaded / p.total
		fmt.Fprintf(os.Stderr, "\rDownloading %s %3d%%", p.name, pct)
	} else {
		fmt.Fprintf(os.Stderr, "\rDownloading %s %d MB", p.name, p.downloaded/1e6)
	}
}

func (p *progressWriter) finish(ok bool) {
	if ok && p.total > 0 {
		fmt.Fprintf(os.Stderr, "\rDownloading %s 100%%\n", p.name)
	} else if ok {
		fmt.Fprintf(os.Stderr, "\rDownloading %s done\n", p.name)
	} else {
		fmt.Fprintln(os.Stderr)
	}
}

// finalizeFile создаёт symlink snapshots/... -> blobs/{oid} (или copy при ошибке)
func finalizeFile(f FileInfo) (string, error) {
	if f.LocalPath == f.FinalPath {
		return f.FinalPath, nil
	}

	if _, err := os.Stat(f.FinalPath); err == nil {
		return f.FinalPath, nil
	}

	if _, err := os.Stat(f.LocalPath); err != nil {
		return f.FinalPath, fmt.Errorf("blob отсутствует: %s", f.LocalPath)
	}

	if err := os.MkdirAll(filepath.Dir(f.FinalPath), 0o755); err != nil {
		return "", err
	}

	symlinkMu.Lock()
	disabled := symlinksDisabled
	symlinkMu.Unlock()

	if !disabled {
		rel, err := filepath.Rel(filepath.Dir(f.FinalPath), f.LocalPath)
		if err == nil {
			if err := os.Symlink(rel, f.FinalPath); err == nil {
				return f.FinalPath, nil
			}
		}

		symlinkMu.Lock()
		if !symlinksDisabled {
			symlinksDisabled = true
			fmt.Fprintln(os.Stderr, "hf: symlink недоступен, копирование в snapshots")
		}

		symlinkMu.Unlock()
	}

	// degraded: hardlink или copy
	if err := os.Link(f.LocalPath, f.FinalPath); err == nil {
		return f.FinalPath, nil
	}

	src, err := os.Open(f.LocalPath)
	if err != nil {
		return "", err
	}
	defer src.Close()

	dst, err := os.Create(f.FinalPath)
	if err != nil {
		return "", err
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		_ = os.Remove(f.FinalPath)
		return "", err
	}

	return f.FinalPath, nil
}

// ensureDownloaded скачивает файл если его ещё нет в кэше
func ensureDownloaded(f FileInfo, token string) (string, error) {
	if _, err := os.Stat(f.FinalPath); err == nil {
		return f.FinalPath, nil
	}

	if f.OID != "" {
		if _, err := os.Stat(f.LocalPath); err == nil {
			return finalizeFile(f)
		}
	} else {
		// маленький файл без LFS oid - качаем прямо в FinalPath
		f.LocalPath = f.FinalPath
	}

	fmt.Fprintf(os.Stderr, "hf: скачивание %s\n", f.Path)

	if err := downloadFile(f.URL, f.LocalPath, token, f.Size); err != nil {
		return "", err
	}

	if f.LocalPath == f.FinalPath {
		return f.FinalPath, nil
	}

	return finalizeFile(f)
}
