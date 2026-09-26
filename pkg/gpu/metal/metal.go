// Package metal - каркас Metal-backend для macOS.
//
// Статус: только scaffold.
// Структура Backend реализует интерфейс gpu.Backend целиком, но все вычисления возвращают ErrUnavailable - Metal-kernels ещё нет.
// Смысл пакета: закрепить точку расширения (gpu.OpenMetal) и не ломать CUDA-путь.
//
// Пакет намеренно не импортирует pkg/gpu (иначе цикл импорта): Open возвращает конкретный *Backend, а gpu.OpenMetal приводит его к gpu.Backend.
package metal

import (
	"errors"

	"github.com/magomedcoder/gogguf/pkg/format"
	"github.com/magomedcoder/gogguf/pkg/ops"
)

// ErrUnavailable: Metal-backend пока не считает ничего
var ErrUnavailable = errors.New("metal: backend не реализован (scaffold)")

// Backend - заглушка устройства Metal: реализует весь интерфейс gpu.Backend, но на любой compute-вызов отдаёт ErrUnavailable
type Backend struct {
	name string
}

// Name возвращает имя устройства, например "Metal (scaffold)"
func (b *Backend) Name() string {
	if b == nil || b.name == "" {
		return "Metal (scaffold)"
	}

	return b.name
}

// VRAMInfo: размер unified memory пока не запрашивается
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

func (b *Backend) FFNSwiGLUQuantCached(format.GGML, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return ErrUnavailable
}

func (b *Backend) FFNGeGLUQuantCached(format.GGML, string, string, string, []byte, []byte, []byte, []float32, []float32, int, int) error {
	return ErrUnavailable
}

func (b *Backend) AttnFFNResidualQuantCached(_ format.GGML, _, _, _, _, _ string, _, _, _, _ []byte, _, _, _ []float32, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

func (b *Backend) QKVRoPEAttentionQuantCached(_ format.GGML, _ ops.RoPEMode, _, _, _, _, _, _ string, _, _, _ []byte, _, _, _, _, _, _, _, _, _ []float32, _, _, _, _, _, _, _ int, _ float32) error {
	return ErrUnavailable
}

// HiddenResident: device-resident hidden state потребует Metal-kernels
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

func (b *Backend) LogitsFromDevice(format.GGML, string, []float32, string, []byte, []float32, []float32, int, int, float32) error {
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
