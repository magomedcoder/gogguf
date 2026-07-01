package runtime

import "testing"

func TestTokensPrefixEqual(t *testing.T) {
	a := []int{1, 2, 3, 4}
	b := []int{1, 2, 5, 6}

	if !tokensPrefixEqual(a, b, 2) {
		t.Fatal("ожидали совпадение префикса длины 2")
	}

	if tokensPrefixEqual(a, b, 3) {
		t.Fatal("не ожидали совпадение префикса длины 3")
	}

	if tokensPrefixEqual(a, b, 5) {
		t.Fatal("не ожидали совпадение при n > len")
	}
}

func TestShouldResetConversation(t *testing.T) {
	cached := []int{1, 2, 3, 4, 5}

	if !shouldResetConversation(len(cached), cached, []int{9, 9}) {
		t.Fatal("ожидали сброс при более коротком промпте")
	}

	if shouldResetConversation(len(cached), cached, []int{1, 2, 3, 4, 5, 6}) {
		t.Fatal("не ожидали сброс при совпадающем префиксе")
	}

	if !shouldResetConversation(len(cached), cached, []int{1, 2, 9, 4, 5}) {
		t.Fatal("ожидали сброс при несовпадающем префиксе")
	}
}
