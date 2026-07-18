package format

import "fmt"

// GGML представляет кодировку данных тензора
type GGML uint32

const (
	GgmlFloat32 GGML = 0
	GgmlFloat16 GGML = 1
	GgmlQ4_0    GGML = 2
	GgmlQ4_1    GGML = 3
	GgmlQ5_0    GGML = 6
	GgmlQ5_1    GGML = 7
	GgmlQ8_0    GGML = 8
	GgmlQ8_1    GGML = 9
	GgmlQ2_K    GGML = 10
	GgmlQ3_K    GGML = 11
	GgmlQ4_K    GGML = 12
	GgmlQ5_K    GGML = 13
	GgmlQ6_K    GGML = 14
	GgmlQ8_K    GGML = 15
	GgmlInt8    GGML = 16
	GgmlInt16   GGML = 17
	GgmlInt32   GGML = 18
)

var ggmlNames = map[GGML]string{
	GgmlFloat32: "float32",
	GgmlFloat16: "float16",
	GgmlQ4_0:    "q4_0",
	GgmlQ4_1:    "q4_1",
	GgmlQ5_0:    "q5_0",
	GgmlQ5_1:    "q5_1",
	GgmlQ8_0:    "q8_0",
	GgmlQ8_1:    "q8_1",
	GgmlQ2_K:    "q2_k",
	GgmlQ3_K:    "q3_k",
	GgmlQ4_K:    "q4_k",
	GgmlQ5_K:    "q5_k",
	GgmlQ6_K:    "q6_k",
	GgmlQ8_K:    "q8_k",
	GgmlInt8:    "int8",
	GgmlInt16:   "int16",
	GgmlInt32:   "int32",
}

const (
	qK4_0      = 32
	qK4_1      = 32
	qK5_0      = 32
	qK5_1      = 32
	qK8_0      = 32
	qK_K       = 256
	kScaleSize = 12
)

type ggmlBlock struct {
	blockSize     uint64
	valuesInBlock uint64
}

var ggmlBlocks = map[GGML]ggmlBlock{
	GgmlFloat32: {blockSize: 4, valuesInBlock: 1},
	GgmlFloat16: {blockSize: 2, valuesInBlock: 1},
	GgmlQ4_0:    {blockSize: 2 + qK4_0/2, valuesInBlock: qK4_0},
	GgmlQ4_1:    {blockSize: 4 + qK4_1/2, valuesInBlock: qK4_1},
	GgmlQ5_0:    {blockSize: 2 + 4 + qK5_0/2, valuesInBlock: qK5_0},
	GgmlQ5_1:    {blockSize: 4 + 4 + qK5_1/2, valuesInBlock: qK5_1},
	GgmlQ8_0:    {blockSize: 2 + qK8_0, valuesInBlock: qK8_0},
	GgmlQ8_1:    {blockSize: 2 + 2 + qK8_0, valuesInBlock: qK8_0},
	GgmlQ2_K:    {blockSize: qK_K/2 + qK_K/4 + 2 + 2, valuesInBlock: qK_K},
	GgmlQ3_K:    {blockSize: qK_K/8 + qK_K/4 + 2 + 2, valuesInBlock: qK_K},
	GgmlQ4_K:    {blockSize: 2*2 + kScaleSize + qK_K/2, valuesInBlock: qK_K},
	GgmlQ5_K:    {blockSize: 2 + 2 + kScaleSize + qK_K/2, valuesInBlock: qK_K},
	GgmlQ6_K:    {blockSize: qK_K/2 + qK_K/4 + qK_K/16 + 2, valuesInBlock: qK_K},
	GgmlQ8_K:    {blockSize: 4 + qK_K + 2*qK_K/16, valuesInBlock: qK_K},
	GgmlInt8:    {blockSize: 1, valuesInBlock: 1},
	GgmlInt16:   {blockSize: 2, valuesInBlock: 1},
	GgmlInt32:   {blockSize: 4, valuesInBlock: 1},
}

// String возвращает имя типа GGML
func (g GGML) String() string {
	if name, ok := ggmlNames[g]; ok {
		return name
	}

	return fmt.Sprintf("неизвестный GGML(%d)", g)
}

// dataSize вычисляет размер данных тензора в байтах
func (g GGML) dataSize(dimensions []uint64) int64 {
	block, ok := ggmlBlocks[g]
	if !ok {
		panic("неизвестный тип: " + g.String())
	}

	values := uint64(1)
	for _, d := range dimensions {
		values *= d
	}

	return int64((values / block.valuesInBlock) * block.blockSize)
}
