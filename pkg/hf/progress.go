package hf

import (
	"io"
	"os"
)

// ProgressWriter receives download/resolve progress lines.
// Default is os.Stderr. Set to io.Discard to silence. Not required for inference.
var ProgressWriter io.Writer = os.Stderr

func progressOut() io.Writer {
	if ProgressWriter == nil {
		return io.Discard
	}

	return ProgressWriter
}
