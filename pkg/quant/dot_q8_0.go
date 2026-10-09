package quant

import "encoding/binary"

// dotBlockQ8_0 - dot product of one Q8_0 block (32 values) with float32 vector
var dotBlockQ8_0 = dotBlockQ8_0Go

func dotBlockQ8_0Go(block []byte, x []float32) float32 {
	d := FP16ToFP32(binary.LittleEndian.Uint16(block[0:2]))
	var sum float32
	for i := range QK8_0 {
		sum += d * float32(int8(block[2+i])) * x[i]
	}

	return sum
}
