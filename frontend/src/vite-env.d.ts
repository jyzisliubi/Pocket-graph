/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API 后端地址，默认 http://localhost:8000 */
  readonly VITE_API_BASE_URL?: string
  /** API Key（请求头 X-API-Key） */
  readonly VITE_API_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
