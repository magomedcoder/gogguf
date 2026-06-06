package gguf

import "fmt"

// Type - тип значения метаданных GGUF
type Type uint32

const (
	Uint8   Type = 0
	Int8    Type = 1
	Uint16  Type = 2
	Int16   Type = 3
	Uint32  Type = 4
	Int32   Type = 5
	Float32 Type = 6
	Bool    Type = 7
	String  Type = 8
	Array   Type = 9
	Uint64  Type = 10
	Int64   Type = 11
	Float64 Type = 12
)

var typeNames = map[Type]string{
	Uint8:   "uint8",
	Int8:    "int8",
	Uint16:  "uint16",
	Int16:   "int16",
	Uint32:  "uint32",
	Int32:   "int32",
	Float32: "float32",
	Bool:    "bool",
	String:  "string",
	Array:   "array",
	Uint64:  "uint64",
	Int64:   "int64",
	Float64: "float64",
}

// String возвращает имя типа метаданных
func (t Type) String() string {
	if name, ok := typeNames[t]; ok {
		return name
	}
	return fmt.Sprintf("неизвестный-тип-%d", t)
}

// Filetype - тип большинства тензоров в файле
type Filetype uint32

const (
	AllF32            Filetype = 0
	MostlyF16         Filetype = 1
	MostlyQ4_0        Filetype = 2
	MostlyQ4_1        Filetype = 3
	MostlyQ4_1SomeF16 Filetype = 4
	MostlyQ8_0        Filetype = 7
	MostlyQ5_0        Filetype = 8
	MostlyQ5_1        Filetype = 9
	MostlyQ2_K        Filetype = 10
	MostlyQ3_KS       Filetype = 11
	MostlyQ3_KM       Filetype = 12
	MostlyQ3_KL       Filetype = 13
	MostlyQ4_KS       Filetype = 14
	MostlyQ4_KM       Filetype = 15
	MostlyQ5_KS       Filetype = 16
	MostlyQ5_KM       Filetype = 17
	MostlyQ6_K        Filetype = 18
)

var filetypeNames = map[Filetype]string{
	AllF32:            "все F32",
	MostlyF16:         "в основном F16",
	MostlyQ4_0:        "в основном Q4_0",
	MostlyQ4_1:        "в основном Q4_1",
	MostlyQ4_1SomeF16: "в основном Q4_1, частично F16",
	MostlyQ8_0:        "в основном Q8_0",
	MostlyQ5_0:        "в основном Q5_0",
	MostlyQ5_1:        "в основном Q5_1",
	MostlyQ2_K:        "в основном Q2_K",
	MostlyQ3_KS:       "в основном Q3_K - малый",
	MostlyQ3_KM:       "в основном Q3_K - средний",
	MostlyQ3_KL:       "в основном Q3_K - большой",
	MostlyQ4_KS:       "в основном Q4_K - малый",
	MostlyQ4_KM:       "в основном Q4_K - средний",
	MostlyQ5_KS:       "в основном Q5_K - малый",
	MostlyQ5_KM:       "в основном Q5_K - средний",
	MostlyQ6_K:        "в основном Q6_K",
}

// String возвращает описание типа квантизации файла
func (f Filetype) String() string {
	if name, ok := filetypeNames[f]; ok {
		return name
	}
	return fmt.Sprintf("неизвестный filetype(%d)", f)
}
