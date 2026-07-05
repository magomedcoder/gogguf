export type Role = 'user' | 'assistant' | 'system'

export interface Message {
  role: Role
  content: string
}

export interface ModelInfo {
  id: string
  name?: string
  architecture?: string
  context_length?: number
  chat_template?: boolean
}

export interface ChatSettings {
  maxTokens: number
  temperature: number
  thinking: boolean
  repeatPenalty: number
  minP: number
}

export interface ChatStreamChunk {
  choices?: Array<{
    delta?: {
      role?: string
      content?: string
    }
  }>
  error?: string
}
