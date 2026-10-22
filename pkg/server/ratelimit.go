package server

import (
	"net/http"
	"strconv"
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
	_, ok := rl.allowWithRemaining(key)
	return ok
}

// allowWithRemaining consumes one token when available and returns remaining whole tokens
func (rl *rateLimiter) allowWithRemaining(key string) (remaining int, ok bool) {
	if rl == nil {
		return 0, true
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
		tokens := rl.burst - 1
		rl.clients[key] = &bucket{
			tokens:   tokens,
			last:     now,
			lastSeen: now,
		}
		if tokens < 0 {
			tokens = 0
		}

		return int(tokens), true
	}

	elapsed := now.Sub(b.last).Seconds()
	b.tokens += elapsed * rl.rate
	if b.tokens > rl.burst {
		b.tokens = rl.burst
	}

	b.last = now
	b.lastSeen = now
	if b.tokens < 1 {
		return 0, false
	}

	b.tokens--
	rem := int(b.tokens)
	if rem < 0 {
		rem = 0
	}

	return rem, true
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
		limit := int(rl.burst + 0.5)
		w.Header().Set("X-RateLimit-Limit-Requests", strconv.Itoa(limit))
		remaining, ok := rl.allowWithRemaining(clientIP(r))
		w.Header().Set("X-RateLimit-Remaining-Requests", strconv.Itoa(remaining))
		w.Header().Set("X-RateLimit-Reset-Requests", "1")
		if !ok {
			w.Header().Set("Retry-After", "1")
			writeAPIError(w, http.StatusTooManyRequests, "rate limit exceeded", "rate_limit_error")
			return
		}

		next.ServeHTTP(w, r)
	})
}
