package metal

import (
	"errors"

	"github.com/magomedcoder/gogguf/pkg/compute"
	"github.com/magomedcoder/gogguf/pkg/dtype"
)

// ErrUnavailable Metal backend does not compute yet
var ErrUnavailable = errors.New("metal: backend not implemented (scaffold)")

// Backend - Metal device stub: implements full device.Backend interface but any compute call returns ErrUnavailable
type Backend struct {
	name string
}

// Name returns device name, e.g. "Metal (scaffold)"
func (b *Backend) Name() string {
	if b == nil || b.name == "" {
		return "Metal (scaffold)"
	}

	return b.name
}

// VRAMInfo: unified memory size not queried yet
func (b *Backend) VRAMInfo() (used, total uint64, err error) {
	return 0, 0, nil
}

func (b *Backend) MatMulVec([]float32, int, int, []float32) ([]float32, error) {
	return nil, ErrUnavailable
}

func (b *Backend) MatMulVecCached(string, []float32, int, int, []float32) ([]float32, error) {
	return nil, ErrUnavailable
}

func (b *Backend) MatMulVecQ8_0Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, ErrUnavailable
}

func (b *Backend) MatMulVecQ4_0Cached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, ErrUnavailable
}

func (b *Backend) MatMulVecQ4_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, ErrUnavailable
}

func (b *Backend) MatMulVecQ5_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, ErrUnavailable
}

func (b *Backend) MatMulVecQ6_KCached(string, []byte, int, int, []float32) ([]float32, error) {
	return nil, ErrUnavailable
}

func (b *Backend) RMSNormInto(_, _, _ []float32, _ float32) error {
	return ErrUnavailable
}

func (b *Backend) ApplyRoPEHeads([]float32, int, int, int, float32) error {
	return ErrUnavailable
}

func (b *Backend) ApplyRoPEHeadsNorm([]float32, int, int, int, float32) error {
	return ErrUnavailable
}

func (b *Backend) SwiGLUInPlace(_, _ []float32) error {
	return ErrUnavailable
}

func (b *Backend) FFNSwiGLUCached(_, _, _ string, _, _, _, _, _ []float32, _, _ int) error {
	return ErrUnavailable
}

func (b *Backend) FFNSwiGLUQ8_0Cached(_, _, _ string, _, _, _ []byte, _, _ []float32, _, _ int) error {
	return ErrUnavailable
}

func (b *Backend) AttnFFNResidualCached(_, _, _, _, _ string, _, _, _, _, _, _, _ []float32, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

func (b *Backend) AttnFFNResidualQ8_0Cached(_, _, _, _, _ string, _, _, _, _ []byte, _, _, _ []float32, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

func (b *Backend) QKVRoPEAttentionCached(_, _, _, _, _ string, _, _, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

func (b *Backend) QKVRoPEAttentionQ8_0Cached(_, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

func (b *Backend) FFNSwiGLUQuantCached(dtype.DType, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return ErrUnavailable
}

func (b *Backend) FFNGeGLUQuantCached(dtype.DType, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return ErrUnavailable
}

func (b *Backend) AttnFFNResidualQuantCached(_ dtype.DType, _, _, _, _, _ string, _, _, _, _ []byte, _, _, _ []float32, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

func (b *Backend) QKVRoPEAttentionQuantCached(_ dtype.DType, _ compute.RoPEMode, _, _, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

// HiddenResident device-resident hidden state will require Metal kernels
func (b *Backend) HiddenResident() bool {
	return false
}

func (b *Backend) HiddenActive() bool {
	return false
}

func (b *Backend) HiddenUpload([]float32) error {
	return ErrUnavailable
}

func (b *Backend) HiddenDownload([]float32) error {
	return ErrUnavailable
}

func (b *Backend) LogitsFromDevice(dtype.DType, string, []float32, string, []byte, []float32, []float32, int, int, float32) error {
	return ErrUnavailable
}

func (b *Backend) AttentionScoresInto(_, _, _, _, _ []float32, _, _, _, _ int) error {
	return ErrUnavailable
}

func (b *Backend) KVCacheInit(int, int, int, int, int) error {
	return ErrUnavailable
}

func (b *Backend) KVCacheReset() {}

func (b *Backend) KVCacheAppend(int, int, []float32, []float32) error {
	return ErrUnavailable
}

func (b *Backend) KVCacheAppendN(int, int, []float32, []float32, int) error {
	return ErrUnavailable
}

func (b *Backend) AttentionScoresKV(int, []float32, []float32, int, int, int, int) error {
	return ErrUnavailable
}

func (b *Backend) Close() error {
	return nil
}
