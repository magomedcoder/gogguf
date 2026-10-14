//go:build linux

package weights

import (
	"os"
	"syscall"
	"unsafe"
)

// adviseDontNeed drops resident pages fully covered by view (page-aligned inner range).
// view must be a file-backed mmap slice (caller checks via MappedRawView).
func adviseDontNeed(view []byte) int64 {
	if len(view) == 0 {
		return 0
	}

	page := uintptr(os.Getpagesize())
	start := uintptr(unsafe.Pointer(&view[0]))
	end := start + uintptr(len(view))

	// only whole pages inside the tensor: neighbours sharing a boundary page keep it
	alignedStart := (start + page - 1) &^ (page - 1)
	alignedEnd := end &^ (page - 1)
	if alignedEnd <= alignedStart {
		return 0
	}

	n := int(alignedEnd - alignedStart)
	off := int(alignedStart - start)
	b := view[off : off+n]
	if err := syscall.Madvise(b, syscall.MADV_DONTNEED); err != nil {
		return 0
	}

	return int64(n)
}
