import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

export type Theme = 'dark' | 'light'

type ThemeProviderProps = {
  children: ReactNode
  /** 默认主题，未从 localStorage 读取到时使用 */
  defaultTheme?: Theme
  /** localStorage 存储键名 */
  storageKey?: string
}

type ThemeProviderState = {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

const initialState: ThemeProviderState = {
  theme: 'dark',
  setTheme: () => null,
  toggleTheme: () => null,
}

const ThemeProviderContext = createContext<ThemeProviderState>(initialState)

/**
 * 主题 Provider：管理 Dark/Light 模式，使用 localStorage 持久化。
 * 默认 Dark 模式。通过给 <html> 元素切换 class 来控制 Tailwind dark 模式。
 */
export function ThemeProvider({
  children,
  defaultTheme = 'dark',
  storageKey = 'pocketgraphrag-theme',
}: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === 'undefined') return defaultTheme
    const stored = window.localStorage.getItem(storageKey) as Theme | null
    return stored ?? defaultTheme
  })

  useEffect(() => {
    const root = window.document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(theme)
    root.style.colorScheme = theme
  }, [theme])

  const setTheme = (next: Theme) => {
    window.localStorage.setItem(storageKey, next)
    setThemeState(next)
  }

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark')

  return (
    <ThemeProviderContext.Provider value={{ theme, setTheme, toggleTheme }}>
      {children}
    </ThemeProviderContext.Provider>
  )
}

/**
 * 获取当前主题与切换方法
 */
export function useTheme() {
  const context = useContext(ThemeProviderContext)
  if (context === undefined)
    throw new Error('useTheme 必须在 ThemeProvider 内部使用')
  return context
}
