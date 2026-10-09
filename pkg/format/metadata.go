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
		return zero, fmt.Errorf("значение метаданных %q не найдено", name)
	}

	if _, ok := v.(T); !ok {
		return zero, fmt.Errorf("значение метаданных %q не является типом %T, фактический тип: %T", name, zero, v)
	}

	return v.(T), nil
}

// MetaValueNumber returns metadata value with given name as number
// If value is not a number, returns error
// Number is converted to type T
// Useful when exact numeric type does not matter
func MetaValueNumber[T ~int | ~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64](metadata Metadata, name string) (T, error) {
	v, found := metadata[name]
	if !found {
		return 0, fmt.Errorf("значение метаданных %q не найдено", name)
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
		return 0, fmt.Errorf("значение метаданных %q не является числом, тип: %T", name, v)
	}
}
