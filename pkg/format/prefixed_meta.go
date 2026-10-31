package format

// PrefixedMeta reads metadata keys under a fixed prefix (e.g. "llama.")
type PrefixedMeta struct {
	Prefix string
	MD     Metadata
}

// Key returns the full metadata key for a suffix
func (p PrefixedMeta) Key(key string) string {
	return p.Prefix + key
}

// Int returns the metadata value as int
func (p PrefixedMeta) Int(key string) (int, error) {
	return p.MD.Int(p.Key(key))
}

// IntOr returns the metadata value as int, or def on error
func (p PrefixedMeta) IntOr(key string, def int) int {
	return MetaValueNumberOr[int](p.MD, p.Key(key), def)
}

// Float32Or returns the metadata value as float32, or def on error
func (p PrefixedMeta) Float32Or(key string, def float32) float32 {
	return MetaValueOr[float32](p.MD, p.Key(key), def)
}
