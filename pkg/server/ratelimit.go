package server

import (
	"net/http"
	"sync"
	"time"
)

// rateLimiter is a simple token bucket keyed by client (usually IP).
type rateLimiter struct {
	mu        sync.Mutex
	rate      float64 // tokens per second
	burst     float64
	clients   map[string]*bucket
	lastSweep time.Time
}

type bucket struct {
	tokens   float64
	last     time.Time
	lastSeen time.Time
}

func newRateLimiter(perMinute int) *rateLimiter {
	if perMinute < 1 {
		return nil
	}

	rate := float64(perMinute) / 60.0
	burst := float64(perMinute)
	if burst < 1 {
		burst = 1
	}

	return &rateLimiter{
		rate:      rate,
		burst:     burst,
		clients:   make(map[string]*bucket),
		lastSweep: time.Now(),
	}
}

func (rl *rateLimiter) allow(key string) bool {
	if rl == nil {
		return true
	}

	now := time.Now()
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if now.Sub(rl.lastSweep) > 5*time.Minute {
		rl.sweepLocked(now)
		rl.lastSweep = now
	}

	b := rl.clients[key]
	if b == nil {
		rl.clients[key] = &bucket{
			tokens:   rl.burst - 1,
			last:     now,
			lastSeen: now,
		}
		return true
	}
	elapsed := now.Sub(b.last).Seconds()
	b.tokens += elapsed * rl.rate
	if b.tokens > rl.burst {
		b.tokens = rl.burst
	}

	b.last = now
	b.lastSeen = now
	if b.tokens < 1 {
		return false
	}

	b.tokens--

	return true
}

func (rl *rateLimiter) sweepLocked(now time.Time) {
	for k, b := range rl.clients {
		if now.Sub(b.lastSeen) > 10*time.Minute {
			delete(rl.clients, k)
		}
	}
}

func withRateLimit(next http.Handler, rl *rateLimiter) http.Handler {
	if rl == nil {
		return next
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rl.allow(clientIP(r)) {
			w.Header().Set("Retry-After", "1")
			writeAPIError(w, http.StatusTooManyRequests, "rate limit exceeded", "rate_limit_error")
			return
		}

		next.ServeHTTP(w, r)
	})
}
