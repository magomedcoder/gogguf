package qwen3

import "github.com/magomedcoder/gogguf/pkg/debug"

// DebugHooks is a type alias of [debug.Hooks] for backward-compatible API surface in this architecture package.
// Prefer [debug.Hooks] and [model.DebugHookSetter] in new code.
type DebugHooks = debug.Hooks
