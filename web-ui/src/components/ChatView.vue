<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { fetchModels, resetConversation, streamChat } from '@/api'
import { APP_LOCALES, setLocale } from '@/i18n'
import type { Message, ModelInfo } from '@/types'

const { t, locale } = useI18n()

const messages = ref<Message[]>([])
const input = ref('')
const loading = ref(false)
const error = ref('')
const model = ref<ModelInfo | null>(null)
const messagesEl = ref<HTMLElement | null>(null)

const settings = reactive({
  maxTokens: 512,
  temperature: 0.7,
  thinking: false,
})

const modelTitle = computed(() => model.value?.name || model.value?.id || 'gguf.go')
const modelMeta = computed(() => {
  const parts: string[] = []
  if (model.value?.architecture) {
    parts.push(model.value.architecture)
  }

  if (model.value?.context_length) {
    parts.push(t('meta.contextLength', { n: model.value.context_length }))
  }

  return parts.join(' · ')
})
const canSend = computed(() => input.value.trim().length > 0 && !loading.value)
const streamingIndex = computed(() => loading.value && messages.value.length > 0 ? messages.value.length - 1 : -1)

let abortController: AbortController | null = null

async function scrollToBottom() {
  await nextTick()
  const el = messagesEl.value
  if (el) {
    el.scrollTop = el.scrollHeight
  }
}

async function loadModel() {
  try {
    const models = await fetchModels()
    model.value = models[0] ?? null
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

async function newChat() {
  stopGeneration()
  messages.value = []
  input.value = ''
  error.value = ''
  try {
    await resetConversation()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

function stopGeneration() {
  abortController?.abort()
  abortController = null
  loading.value = false
}

async function sendMessage() {
  const text = input.value.trim()
  if (!text || loading.value) {
    return
  }

  error.value = ''
  messages.value.push({
    role: 'user',
    content: text
  })
  input.value = ''
  await scrollToBottom()

  const assistantIndex = messages.value.length
  messages.value.push({
    role: 'assistant',
    content: ''
  })
  loading.value = true
  abortController = new AbortController()

  try {
    const history = messages.value.slice(0, assistantIndex).filter((m) => m.role === 'user' || m.role === 'assistant' || m.role === 'system')

    for await (const token of streamChat(history, settings, abortController.signal)) {
      const msg = messages.value[assistantIndex]
      if (msg) {
        msg.content += token
      }

      await scrollToBottom()
    }

    const finalMsg = messages.value[assistantIndex]
    if (finalMsg && !finalMsg.content) {
      messages.value.splice(assistantIndex, 1)
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      const finalMsg = messages.value[assistantIndex]
      if (finalMsg && !finalMsg.content) {
        messages.value.splice(assistantIndex, 1)
      }
    } else {
      messages.value.splice(assistantIndex, 1)
      error.value = e instanceof Error ? e.message : String(e)
    }
  } finally {
    loading.value = false
    abortController = null
    await scrollToBottom()
  }
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void sendMessage()
  }
}

onMounted(() => {
  void loadModel()
})

watch(messages, () => {
  void scrollToBottom()
}, { deep: true })
</script>

<template>
  <div class="mx-auto flex min-h-screen max-w-[720px] flex-col px-5">
    <header class="flex items-start justify-between gap-4 border-b border-[#2a2a2a] py-5 pb-3.5">
      <div class="min-w-0">
        <h1 class="truncate text-base font-semibold">{{ modelTitle }}</h1>
        <p v-if="modelMeta" class="mt-1 text-xs text-[#888]">{{ modelMeta }}</p>
      </div>
      <div class="flex shrink-0 items-center gap-3">
        <div class="flex gap-1 text-xs">
          <button
            v-for="{ code, label } in APP_LOCALES"
            :key="code"
            type="button"
            class="rounded px-1.5 py-0.5 border-none bg-transparent"
            :class="locale === code ? 'text-[#e8e8e8]' : 'text-[#888] hover:text-[#e8e8e8]'"
            @click="setLocale(code)"
          >
            {{ label }}
          </button>
        </div>
        <button
          type="button"
          class="border-none bg-transparent text-sm text-[#888] hover:text-[#e8e8e8]"
          @click="newChat"
        >
          {{ t('chat.newChat') }}
        </button>
      </div>
    </header>

    <details class="border-b border-[#2a2a2a] py-2.5 text-xs text-[#888] open:[&>summary]:mb-2.5">
      <summary class="cursor-pointer select-none">{{ t('chat.settings') }}</summary>
      <div class="flex flex-wrap gap-x-5 gap-y-3">
        <label class="inline-flex items-center gap-2">
          {{ t('settings.maxTokens') }}
          <input
            v-model.number="settings.maxTokens"
            type="number"
            min="1"
            max="4096"
            class="w-[72px] rounded-md border border-[#2a2a2a] bg-[#0f0f0f] px-2 py-1 text-[#e8e8e8]"
          />
        </label>
        <label class="inline-flex items-center gap-2">
          {{ t('settings.temperature') }}
          <input
            v-model.number="settings.temperature"
            type="number"
            min="0"
            max="2"
            step="0.1"
            class="w-[72px] rounded-md border border-[#2a2a2a] bg-[#0f0f0f] px-2 py-1 text-[#e8e8e8]"
          />
        </label>
        <label class="inline-flex items-center gap-1.5 text-[#e8e8e8]">
          <input
            v-model="settings.thinking"
            type="checkbox"
            class="accent-[#6ea8fe]"
          />
          {{ t('settings.thinking') }}
        </label>
      </div>
    </details>

    <main ref="messagesEl" class="flex flex-1 flex-col gap-3.5 overflow-y-auto py-5">
      <p
        v-if="messages.length === 0"
        class="m-auto text-center text-sm text-[#888]"
      >
        {{ t('chat.emptyMessage') }}
      </p>

      <div
        v-for="(message, index) in messages"
        :key="index"
        class="max-w-full whitespace-pre-wrap break-words leading-relaxed"
        :class="message.role === 'user'
          ? 'max-w-[80%] self-end rounded-xl rounded-br rounded-bl-sm border border-[#2a2a2a] bg-[#1c1c1c] px-3.5 py-2.5'
          : 'self-start rounded-xl rounded-bl rounded-br-sm border border-[#2a2a2a] px-3.5 py-2.5 text-[#ccc]'"
      >
        {{ message.content }}<span
          v-if="index === streamingIndex"
          class="animate-blink text-[#6ea8fe]"
        >| </span>
      </div>
    </main>

    <p v-if="error" class="mb-2 text-sm text-[#ee5555]">{{ error }}</p>

    <footer class="sticky bottom-0 border-t border-[#2a2a2a] bg-[#0f0f0f] py-3.5 pb-5">
      <textarea
        v-model="input"
        rows="2"
        :placeholder="t('chat.placeholder')"
        :disabled="loading"
        class="w-full resize-none rounded-xl border border-[#2a2a2a] bg-[#141414] px-3.5 py-3 text-[#e8e8e8] placeholder:text-[#888] focus:border-[#6ea8fe] focus:outline-none disabled:opacity-60"
        @keydown="onKeydown"
      />
      <div class="mt-2.5 flex justify-end gap-2">
        <button
          v-if="loading"
          type="button"
          class="rounded-lg border border-[#ee5555] bg-[#1a1a1a] px-4 py-2 text-[#ee5555]"
          @click="stopGeneration"
        >
          {{ t('chat.stop') }}
        </button>
        <button
          type="button"
          class="rounded-lg border border-[#6ea8fe] bg-[#6ea8fe] px-4 py-2 font-medium text-[#0d1117] hover:border-[#8bb9ff] hover:bg-[#8bb9ff] disabled:cursor-not-allowed disabled:opacity-40"
          :disabled="!canSend"
          @click="sendMessage"
        >
          {{ t('chat.send') }}
        </button>
      </div>
    </footer>
  </div>
</template>

<style scoped>

</style>