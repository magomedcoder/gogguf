package server

import (
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

func TestRateLimiterRefillSynctest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rl := newRateLimiter(60) // 1 token/sec, burst 60
		if rl == nil {
			t.Fatal("nil limiter")
		}

		for i := range 60 {
			if !rl.allow("ip") {
				t.Fatalf("unexpected deny at %d", i)
			}
		}
		if rl.allow("ip") {
			t.Fatal("expected deny after burst")
		}

		time.Sleep(2 * time.Second)
		if !rl.allow("ip") {
			t.Fatal("expected allow after refill")
		}
	})
}

func TestRateLimiterConcurrentSynctest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rl := newRateLimiter(120)
		var wg sync.WaitGroup
		var denied atomic.Int32
		for range 200 {
			wg.Go(func() {
				if !rl.allow("same") {
					denied.Add(1)
				}
			})
		}
		wg.Wait()
		if denied.Load() == 0 {
			t.Fatal("expected some denials under concurrent burst")
		}
	})
}
