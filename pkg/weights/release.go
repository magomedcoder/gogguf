package weights

import "strings"

// Lazy loading model
//
// Tensor bytes returned by Store.Raw are zero-copy views of the mmap'd GGUF file (format.OpenFileMapped): the OS page-faults a tensor in on first touch, so nothing is read from disk at Load time.
// Resident pages stay in RSS / page cache until the kernel reclaims them.
//
// ReleaseRaw / ReleaseLayer give the caller a "layer finished" hook for CPU-only runs: on Linux they madvise(MADV_DONTNEED) the tensor's pages (file-backed MAP_SHARED, so the next access simply re-faults the same bytes - slices already handed out stay valid) and drop the dequantized Floats cache entry.
// Heap copies (non-mmap sources) and non-Linux platforms are left untouched (no-op for the madvise part).

// ReleaseRaw releases resident pages of one tensor and drops its cached Floats copy.
// Returns the number of bytes handed back to the OS (0 when not mmap-backed / unsupported).
func (s *Store) ReleaseRaw(name string) int64 {
	delete(s.floats, name)

	info, err := s.reader.TensorInfo(name)
	if err != nil {
		return 0
	}

	view, ok := info.MappedRawView()
	if !ok {
		return 0
	}

	return adviseDontNeed(view)
}

// ReleaseLayer releases all tensors whose name starts with prefix (e.g. "blk.7.").
// Returns total bytes handed back to the OS.
func (s *Store) ReleaseLayer(prefix string) int64 {
	var total int64
	for i := range s.reader.Tensors {
		name := s.reader.Tensors[i].Name
		if strings.HasPrefix(name, prefix) {
			total += s.ReleaseRaw(name)
		}
	}

	return total
}
