import type { LocaleDefinition } from '../types'

const ru: LocaleDefinition<'ru'> = {
  code: 'ru',
  label: 'Русский',
  messages: {
    chat: {
      newChat: 'новый чат',
      settings: 'настройки',
      emptyMessage: 'Напишите сообщение, чтобы начать диалог',
      placeholder: 'Напишите сообщение...',
      stop: 'Стоп',
      send: 'Отправить',
    },
    settings: {
      maxTokens: 'max_tokens',
      temperature: 'temperature',
      thinking: 'thinking',
      repeatPenalty: 'repeat_penalty',
      minP: 'min_p',
    },
    errors: {
      streamingNotSupported: 'стриминг не поддерживается',
    },
    meta: {
      contextLength: '{n} ctx',
    },
  },
}

export default ru
