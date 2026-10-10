package format

import "fmt"

// Type - GGUF metadata value type
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

// String returns metadata type name
func (t Type) String() string {
	if name, ok := typeNames[t]; ok {
		return name
	}
	return fmt.Sprintf("unknown-type-%d", t)
}

// Filetype - type of majority of tensors in file
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
	AllF32:            "all F32",
	MostlyF16:         "mostly F16",
	MostlyQ4_0:        "mostly Q4_0",
	MostlyQ4_1:        "mostly Q4_1",
	MostlyQ4_1SomeF16: "mostly Q4_1, partially F16",
	MostlyQ8_0:        "mostly Q8_0",
	MostlyQ5_0:        "mostly Q5_0",
	MostlyQ5_1:        "mostly Q5_1",
	MostlyQ2_K:        "mostly Q2_K",
	MostlyQ3_KS:       "mostly Q3_K - small",
	MostlyQ3_KM:       "mostly Q3_K - medium",
	MostlyQ3_KL:       "mostly Q3_K - large",
	MostlyQ4_KS:       "mostly Q4_K - small",
	MostlyQ4_KM:       "mostly Q4_K - medium",
	MostlyQ5_KS:       "mostly Q5_K - small",
	MostlyQ5_KM:       "mostly Q5_K - medium",
	MostlyQ6_K:        "mostly Q6_K",
}

// String returns file quantization type description
func (f Filetype) String() string {
	if name, ok := filetypeNames[f]; ok {
		return name
	}
	return fmt.Sprintf("unknown filetype(%d)", f)
}
