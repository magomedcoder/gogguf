package model

import "github.com/magomedcoder/gogguf/pkg/debug"

// DebugHookSetter is implemented by architectures that expose forward-pass debug callbacks.
type DebugHookSetter interface {
	SetDebugHooks(h *debug.Hooks)
}
