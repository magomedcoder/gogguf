import type { LocaleDefinition } from '../types'

const en: LocaleDefinition<'en'> = {
  code: 'en',
  label: 'English',
  messages: {
    chat: {
      newChat: 'New chat',
      settings: 'Settings',
      emptyMessage: 'Type a message to start a conversation',
      placeholder: 'Type a message...',
      stop: 'Stop',
      send: 'Send',
    },
    settings: {
      maxTokens: 'max_tokens',
      temperature: 'temperature',
      frequencyPenalty: 'frequency_penalty',
    },
    errors: {
      streamingNotSupported: 'Streaming is not supported',
    },
    meta: {
      contextLength: '{n} ctx',
    },
  },
}

export default en
