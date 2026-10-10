package runtime

import "testing"

func TestTokensPrefixEqual(t *testing.T) {
	a := []int{1, 2, 3, 4}
	b := []int{1, 2, 5, 6}

	if !tokensPrefixEqual(a, b, 2) {
		t.Fatal("expected prefix match of length 2")
	}

	if tokensPrefixEqual(a, b, 3) {
		t.Fatal("did not expect prefix match of length 3")
	}

	if tokensPrefixEqual(a, b, 5) {
		t.Fatal("did not expect match when n > len")
	}
}

func TestShouldResetConversation(t *testing.T) {
	cached := []int{1, 2, 3, 4, 5}

	if !shouldResetConversation(len(cached), cached, []int{9, 9}) {
		t.Fatal("expected reset on shorter prompt")
	}

	if shouldResetConversation(len(cached), cached, []int{1, 2, 3, 4, 5, 6}) {
		t.Fatal("did not expect reset on matching prefix")
	}

	if !shouldResetConversation(len(cached), cached, []int{1, 2, 9, 4, 5}) {
		t.Fatal("expected reset on non-matching prefix")
	}
}
