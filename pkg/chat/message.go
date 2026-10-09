package chat

// Message is one dialog message for the chat template.
type Message struct {
	Role       string
	Content    string
	Name       string
	ToolCallID string
	ToolCalls  []ToolCall
}
