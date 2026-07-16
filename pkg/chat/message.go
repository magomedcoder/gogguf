package chat

// Message - одно сообщение диалога для chat template
type Message struct {
	Role       string
	Content    string
	Name       string
	ToolCallID string
	ToolCalls  []ToolCall
}
