package server

import (
	"crypto/subtle"
	"net"
	"net/http"
	"strings"
)

// withAPIKey requires Authorization: Bearer <key> or X-API-Key, except for /v1/health.
func withAPIKey(next http.Handler, apiKey string) http.Handler {
	if apiKey == "" {
		return next
	}

	want := []byte(apiKey)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/health" {
			next.ServeHTTP(w, r)
			return
		}

		got := apiKeyFromRequest(r)
		if subtle.ConstantTimeCompare([]byte(got), want) != 1 {
			writeAPIError(w, http.StatusUnauthorized, "неверный или отсутствующий API key", "authentication_error")
			return
		}

		next.ServeHTTP(w, r)
	})
}

func apiKeyFromRequest(r *http.Request) string {
	if v := strings.TrimSpace(r.Header.Get("X-API-Key")); v != "" {
		return v
	}

	auth := r.Header.Get("Authorization")
	if len(auth) >= 7 && strings.EqualFold(auth[:7], "Bearer ") {
		return strings.TrimSpace(auth[7:])
	}

	return ""
}

func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		if ip := strings.TrimSpace(parts[0]); ip != "" {
			return ip
		}
	}

	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}

	return host
}
