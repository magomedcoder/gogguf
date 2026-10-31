package format

import "fmt"

// Metadata - metadata container in GGUF file
// Values map to corresponding Go types
type Metadata map[string]any

// Int returns metadata value with given name as int
// If value cannot be represented as int, returns error
func (m Metadata) Int(name string) (int, error) {
	return MetaValueNumber[int](m, name)
}

// Any returns metadata value with given name as interface{}
func (m Metadata) Any(name string) (any, error) {
	return MetaValue[any](m, name)
}

// String returns metadata value with given name as string
// If value is not a string, returns error
func (m Metadata) String(name string) (string, error) {
	return MetaValue[string](m, name)
}

// StringArray returns string array from metadata
func (m Metadata) StringArray(name string) ([]string, error) {
	return MetaValue[[]string](m, name)
}

// IntOptional returns int from metadata or defaultVal if key missing
func (m Metadata) IntOptional(name string, defaultVal int) int {
	v, err := m.Int(name)
	if err != nil {
		return defaultVal
	}

	return v
}

// MetaValue returns metadata value with given name as type T
// If value is not T, returns error
func MetaValue[T any](metadata Metadata, name string) (T, error) {
	var zero T
	v, found := metadata[name]
	if !found {
		return zero, fmt.Errorf("metadata value %q not found", name)
	}

	if _, ok := v.(T); !ok {
		return zero, fmt.Errorf("metadata value %q is not type %T, actual type: %T", name, zero, v)
	}

	return v.(T), nil
}

// MetaValueOr returns metadata value as type T, or def if missing or wrong type.
func MetaValueOr[T any](md Metadata, key string, def T) T {
	v, err := MetaValue[T](md, key)
	if err != nil {
		return def
	}

	return v
}

// MetaValueNumber returns metadata value with given name as number
// If value is not a number, returns error
// Number is converted to type T
// Useful when exact numeric type does not matter
func MetaValueNumber[T ~int | ~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64](metadata Metadata, name string) (T, error) {
	v, found := metadata[name]
	if !found {
		return 0, fmt.Errorf("metadata value %q not found", name)
	}

	switch vv := v.(type) {
	case int:
		return T(vv), nil

	case uint8:
		return T(vv), nil

	case int8:
		return T(vv), nil

	case uint16:
		return T(vv), nil

	case int16:
		return T(vv), nil

	case uint32:
		return T(vv), nil

	case int32:
		return T(vv), nil

	case uint64:
		return T(vv), nil

	case int64:
		return T(vv), nil

	case float32:
		return T(vv), nil

	case float64:
		return T(vv), nil

	default:
		return 0, fmt.Errorf("metadata value %q is not a number, type: %T", name, v)
	}
}

// MetaValueNumberOr returns metadata value as number type T, or def on error.
func MetaValueNumberOr[T ~int | ~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64](md Metadata, key string, def T) T {
	v, err := MetaValueNumber[T](md, key)
	if err != nil {
		return def
	}

	return v
}

// VocabSizeFromEmb infers vocab from token_embd.weight dims given embedding length.
func VocabSizeFromEmb(r *Reader, emb int, archLabel string) (int, error) {
	info, err := r.TensorInfo("token_embd.weight")
	if err != nil {
		return 0, err
	}

	if len(info.Dimensions) != 2 {
		return 0, fmt.Errorf("%s: token_embd.weight: expected 2D", archLabel)
	}

	a, b := int(info.Dimensions[0]), int(info.Dimensions[1])
	if a == emb {
		return b, nil
	}

	if b == emb {
		return a, nil
	}

	return 0, fmt.Errorf("%s: token_embd.weight %v does not contain embedding_length=%d", archLabel, info.Dimensions, emb)
}
