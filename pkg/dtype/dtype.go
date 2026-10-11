package dtype

import "fmt"

// DType is the tensor element / quant encoding (GGUF type id)
type DType uint32

const (
	F32   DType = 0
	F16   DType = 1
	Q4_0  DType = 2
	Q4_1  DType = 3
	Q5_0  DType = 6
	Q5_1  DType = 7
	Q8_0  DType = 8
	Q8_1  DType = 9
	Q2_K  DType = 10
	Q3_K  DType = 11
	Q4_K  DType = 12
	Q5_K  DType = 13
	Q6_K  DType = 14
	Q8_K  DType = 15
	Int8  DType = 16
	Int16 DType = 17
	Int32 DType = 18
)

var names = map[DType]string{
	F32:   "float32",
	F16:   "float16",
	Q4_0:  "q4_0",
	Q4_1:  "q4_1",
	Q5_0:  "q5_0",
	Q5_1:  "q5_1",
	Q8_0:  "q8_0",
	Q8_1:  "q8_1",
	Q2_K:  "q2_k",
	Q3_K:  "q3_k",
	Q4_K:  "q4_k",
	Q5_K:  "q5_k",
	Q6_K:  "q6_k",
	Q8_K:  "q8_k",
	Int8:  "int8",
	Int16: "int16",
	Int32: "int32",
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

type blockInfo struct {
	blockSize     uint64
	valuesInBlock uint64
}

var blocks = map[DType]blockInfo{
	F32:   {blockSize: 4, valuesInBlock: 1},
	F16:   {blockSize: 2, valuesInBlock: 1},
	Q4_0:  {blockSize: 2 + qK4_0/2, valuesInBlock: qK4_0},
	Q4_1:  {blockSize: 4 + qK4_1/2, valuesInBlock: qK4_1},
	Q5_0:  {blockSize: 2 + 4 + qK5_0/2, valuesInBlock: qK5_0},
	Q5_1:  {blockSize: 4 + 4 + qK5_1/2, valuesInBlock: qK5_1},
	Q8_0:  {blockSize: 2 + qK8_0, valuesInBlock: qK8_0},
	Q8_1:  {blockSize: 2 + 2 + qK8_0, valuesInBlock: qK8_0},
	Q2_K:  {blockSize: qK_K/16 + qK_K/4 + 2 + 2, valuesInBlock: qK_K},
	Q3_K:  {blockSize: qK_K/8 + qK_K/4 + 12 + 2, valuesInBlock: qK_K},
	Q4_K:  {blockSize: 2*2 + kScaleSize + qK_K/2, valuesInBlock: qK_K},
	Q5_K:  {blockSize: 2*2 + kScaleSize + qK_K/8 + qK_K/2, valuesInBlock: qK_K},
	Q6_K:  {blockSize: qK_K/2 + qK_K/4 + qK_K/16 + 2, valuesInBlock: qK_K},
	Q8_K:  {blockSize: 4 + qK_K + 2*qK_K/16, valuesInBlock: qK_K},
	Int8:  {blockSize: 1, valuesInBlock: 1},
	Int16: {blockSize: 2, valuesInBlock: 1},
	Int32: {blockSize: 4, valuesInBlock: 1},
}

// String returns DType type name.
func (t DType) String() string {
	if name, ok := names[t]; ok {
		return name
	}

	return fmt.Sprintf("unknown DType(%d)", t)
}

// DataSize computes tensor data size in bytes for the given dimensions.
func (t DType) DataSize(dimensions []uint64) int64 {
	block, ok := blocks[t]
	if !ok {
		panic("unknown type: " + t.String())
	}

	values := uint64(1)
	for _, d := range dimensions {
		values *= d
	}

	return int64((values / block.valuesInBlock) * block.blockSize)
}
