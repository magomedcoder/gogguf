import { createI18n } from 'vue-i18n'
import en from './locales/en'
import ru from './locales/ru'
import type { LocaleMessages } from './types'

export const LOCALE_STORAGE_KEY = 'gguf-web-ui-locale'

const localeDefinitions = [en, ru] as const

export type AppLocale = (typeof localeDefinitions)[number]['code']

export const APP_LOCALES = localeDefinitions.map(({ code, label }) => ({ code, label }))

const messages = Object.fromEntries(localeDefinitions.map(({
  code,
  messages: localeMessages
}) => [code, localeMessages])) as Record<AppLocale, LocaleMessages>

const defaultLocale = localeDefinitions.find((locale) => locale.code === 'en')?.code ?? localeDefinitions[0].code

function isAppLocale(value: string): value is AppLocale {
  return localeDefinitions.some((locale) => locale.code === value)
}

function detectLocale(): AppLocale {
  const stored = localStorage.getItem(LOCALE_STORAGE_KEY)
  if (stored && isAppLocale(stored)) {
    return stored
  }

  const browserLocale = navigator.language.toLowerCase()
  const matched = localeDefinitions.find((locale) => browserLocale.startsWith(locale.code))
  return matched?.code ?? defaultLocale
}

export const i18n = createI18n({
  legacy: false,
  locale: detectLocale(),
  fallbackLocale: defaultLocale,
  messages,
})

export function setLocale(locale: AppLocale) {
  i18n.global.locale.value = locale
  localStorage.setItem(LOCALE_STORAGE_KEY, locale)
  document.documentElement.lang = locale
}

document.documentElement.lang = i18n.global.locale.value
