export type LocaleMessages = {
  chat: {
    newChat: string
    settings: string
    emptyMessage: string
    placeholder: string
    stop: string
    send: string
  }
  settings: {
    maxTokens: string
    temperature: string
    thinking: string
    repeatPenalty: string
    minP: string
  }
  errors: {
    streamingNotSupported: string
  }
  meta: {
    contextLength: string
  }
}

export type LocaleDefinition<Code extends string = string> = {
  code: Code
  label: string
  messages: LocaleMessages
}
