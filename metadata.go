package gguf

import "fmt"

// Metadata - контейнер метаданных в файле GGUF
// Значения сопоставляются соответствующим типам Go
type Metadata map[string]interface{}

// Int возвращает значение метаданных с указанным именем как int
// Если значение нельзя представить как int, возвращается ошибка
func (m Metadata) Int(name string) (int, error) {
	return MetaValueNumber[int](m, name)
}

// Any возвращает значение метаданных с указанным именем как interface{}
func (m Metadata) Any(name string) (interface{}, error) {
	return MetaValue[any](m, name)
}

// String возвращает значение метаданных с указанным именем как string
// Если значение не является строкой, возвращается ошибка
func (m Metadata) String(name string) (string, error) {
	return MetaValue[string](m, name)
}

// MetaValue возвращает значение метаданных с указанным именем как тип T
// Если значение не является T, возвращается ошибка
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

// MetaValueNumber возвращает значение метаданных с указанным именем как число
// Если значение не является числом, возвращается ошибка
// Число приводится к типу T
// Полезно, если точный тип числа не важен
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
