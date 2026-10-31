package server

import (
	"sync"
	"testing"
	"testing/synctest"

	"github.com/magomedcoder/gogguf/pkg/runtime"
)

func TestSessionsMapConcurrent(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := New(&runtime.Engine{}, "")
		srv.sessions = map[string]*runtime.Conversation{
			"a": {},
		}

		var wg sync.WaitGroup
		for i := range 64 {
			id := "a"
			if i%2 == 0 {
				id = "b"
			}
			wg.Go(func() {
				srv.sessionsMu.Lock()
				defer srv.sessionsMu.Unlock()
				if _, ok := srv.sessions[id]; !ok {
					srv.sessions[id] = &runtime.Conversation{}
				}
			})
		}
		wg.Wait()

		srv.sessionsMu.Lock()
		defer srv.sessionsMu.Unlock()
		if len(srv.sessions) != 2 {
			t.Fatalf("sessions=%d, want 2", len(srv.sessions))
		}
	})
}
