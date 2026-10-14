//go:build !linux

package weights

// adviseDontNeed is a no-op where madvise is unavailable.
func adviseDontNeed(view []byte) int64 {
	return 0
}
