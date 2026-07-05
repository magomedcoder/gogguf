import { i18n } from '@/i18n'
import type { ChatSettings, ChatStreamChunk, Message, ModelInfo } from './types'

const API_BASE = '/api'

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models`)
  if (!res.ok) {
    throw new Error(await res.text())
  }

  const data = (await res.json()) as {
    models: ModelInfo[]
  }

  return data.models
}

export async function resetConversation(): Promise<void> {
  const res = await fetch(`${API_BASE}/reset`, {
    method: 'POST'
  })
  if (!res.ok) {
    throw new Error(await res.text())
  }
}

export async function* streamChat(
  messages: Message[],
  settings: ChatSettings,
  signal?: AbortSignal,
): AsyncGenerator<string, void, unknown> {
  const res = await fetch(`${API_BASE}/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      stream: true,
      max_tokens: settings.maxTokens,
      temperature: settings.temperature,
      thinking: settings.thinking,
      repeat_penalty: settings.repeatPenalty,
      min_p: settings.minP,
    }),
    signal,
  })

  if (!res.ok) {
    throw new Error(await res.text())
  }

  const reader = res.body?.getReader()
  if (!reader) {
    throw new Error(i18n.global.t('errors.streamingNotSupported'))
  }

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.replace(/\r$/, '').trim()
      if (!trimmed.startsWith('data: ')) {
        continue
      }

      const data = trimmed.slice(6).trim()
      if (data === '[DONE]') {
        return
      }

      const chunk = JSON.parse(data) as ChatStreamChunk
      if (chunk.error) {
        throw new Error(chunk.error)
      }

      const content = chunk.choices?.[0]?.delta?.content
      if (content) {
        yield content
      }
    }
  }
}
