package strutil

import "strings"

// SplitCommaTrim splits s on commas and trims each non-empty part.
func SplitCommaTrim(s string) []string {
	if s == "" {
		return nil
	}

	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}

		out = append(out, p)
	}

	return out
}
