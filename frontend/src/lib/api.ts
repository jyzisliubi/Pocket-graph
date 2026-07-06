import axios, { type AxiosInstance } from 'axios'
import type {
  QARequest,
  QAResponse,
  RetrieveResponse,
  GraphStats,
  SubgraphResponse,
  EntitySearchResult,
  HealthResponse,
  LlmStatusResponse,
  SSEEvent,
  PagerankResponse,
  CommunityResponse,
  ShortestPathResponse,
  DocumentInfo,
  UploadResponse,
  BuildIndexResponse,
  DocumentPreview,
  ImportUrlRequest,
  ImportUrlResponse,
  ExtractSSEEvent,
  MultiModelExtractResponse,
  SettingsResponse,
  SaveSettingsRequest,
  SaveSettingsResponse,
  OllamaModelsResponse,
} from '@/types/api'

/**
 * API 后端地址：优先从环境变量读取，默认指向本地后端
 * 通过 Vite 的 import.meta.env 注入（变量名需以 VITE_ 开头）
 */
const BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/** API Key：从环境变量读取，未配置则为空字符串 */
const API_KEY = import.meta.env.VITE_API_KEY ?? ''

/** axios 实例 */
const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器：自动注入 X-API-Key 头
apiClient.interceptors.request.use((config) => {
  if (API_KEY) {
    config.headers['X-API-Key'] = API_KEY
  }
  return config
})

// 响应拦截器：统一错误处理
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail =
      error?.response?.data?.detail ?? error?.message ?? '请求失败'
    return Promise.reject(new Error(detail))
  },
)

// ==========================
// 健康检查 & LLM 状态
// ==========================

export const getHealth = () =>
  apiClient.get<HealthResponse>('/api/health').then((r) => r.data)

export const getLlmStatus = () =>
  apiClient.get<LlmStatusResponse>('/api/llm/status').then((r) => r.data)

// ==========================
// 系统设置
// ==========================

/** 获取当前 LLM 配置（API Key 已脱敏） */
export const getSettings = () =>
  apiClient.get<SettingsResponse>('/api/settings').then((r) => r.data)

/** 保存 LLM 配置（api_key 为空时后端保持原值） */
export const saveSettings = (data: SaveSettingsRequest) =>
  apiClient.post<SaveSettingsResponse>('/api/settings', data).then((r) => r.data)

/** 获取 Ollama 本地可用模型列表 */
export const getOllamaModels = () =>
  apiClient
    .get<OllamaModelsResponse>('/api/settings/ollama-models')
    .then((r) => r.data)

// ==========================
// 问答接口
// ==========================

/** 非流式问答 */
export const askQuestion = (req: QARequest) =>
  apiClient.post<QAResponse>('/api/qa', req).then((r) => r.data)

/** 只检索不生成 */
export const retrieve = (req: QARequest) =>
  apiClient.post<RetrieveResponse>('/api/retrieve', req).then((r) => r.data)

/**
 * SSE 流式请求通用 helper
 *
 * 通过 fetch 读取 text/event-stream，逐行解析 `data: {...}` 负载，
 * 通过 onEvent 回调把解析后的事件推给调用方。
 *
 * @returns 一个 AbortController，用于主动中断
 */
function streamSSE<TEvent>(
  url: string,
  body: unknown,
  onEvent: (event: TEvent) => void,
  onError?: (err: Error) => void,
): AbortController {
  const controller = new AbortController()

  ;(async () => {
    try {
      const resp = await fetch(`${BASE_URL}${url}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      if (!resp.ok || !resp.body) {
        const text = await resp.text().catch(() => '')
        throw new Error(text || `HTTP ${resp.status}`)
      }

      const reader = resp.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE 以 \n\n 分隔事件块
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() ?? ''

        for (const block of blocks) {
          const line = block.trim()
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (!payload) continue
          try {
            onEvent(JSON.parse(payload) as TEvent)
          } catch {
            // 忽略无法解析的行
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      onError?.(err as Error)
    }
  })()

  return controller
}

/** SSE 流式问答 */
export function askQuestionStream(
  req: QARequest,
  onEvent: (event: SSEEvent) => void,
  onError?: (err: Error) => void,
): AbortController {
  return streamSSE('/api/qa/stream', req, onEvent, onError)
}

// ==========================
// 知识图谱接口
// ==========================

export const getGraphStats = () =>
  apiClient.get<GraphStats>('/api/graph/stats').then((r) => r.data)

export const getTopEntities = (limit = 50) =>
  apiClient
    .get<EntitySearchResult[]>('/api/graph/entities', { params: { limit } })
    .then((r) => r.data)

export const searchEntities = (
  q: string,
  limit = 10,
  threshold = 0.5,
) =>
  apiClient
    .get<EntitySearchResult[]>('/api/graph/search', {
      params: { q, limit, threshold },
    })
    .then((r) => r.data)

export const getEntitySubgraph = (name: string, hops = 1) =>
  apiClient
    .get<SubgraphResponse>(`/api/graph/entity/${encodeURIComponent(name)}/subgraph`, {
      params: { hops },
    })
    .then((r) => r.data)

export const getMultiEntitySubgraph = (entities: string[], hops = 1) =>
  apiClient
    .post<SubgraphResponse>('/api/graph/subgraph', entities, {
      params: { hops },
    })
    .then((r) => r.data)

// ==========================
// 高级图谱算法
// ==========================

/**
 * 获取 PageRank 排序的实体列表
 * 后端 GET /api/graph/pagerank?top_n=N 返回 List[{entity, score}]
 */
export const getPagerank = (top_n = 50) =>
  apiClient
    .get<PagerankResponse[]>('/api/graph/pagerank', { params: { top_n } })
    .then((r) => r.data)

/**
 * 获取社区发现结果列表
 * 后端 GET /api/graph/communities 返回 List[{entity, community_id}]
 */
export const getCommunities = () =>
  apiClient
    .get<CommunityResponse[]>('/api/graph/communities')
    .then((r) => r.data)

/**
 * 查询两实体间的最短路径
 * 后端 GET /api/graph/path?start=&end=&max_hops=N 返回 {path, length}
 */
export const getShortestPath = (start: string, end: string, max_hops = 5) =>
  apiClient
    .get<ShortestPathResponse>('/api/graph/path', {
      params: { start, end, max_hops },
    })
    .then((r) => r.data)

// ==========================
// 文档管理接口
// ==========================

/** 获取文档列表 */
export const getDocuments = () =>
  apiClient.get<DocumentInfo[]>('/api/documents').then((r) => r.data)

/**
 * 上传文档（multipart/form-data）
 * @param file 文件对象
 * @param onProgress 上传进度回调，参数为 0-100 的百分比
 */
export function uploadDocument(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiClient
    .post<UploadResponse>('/api/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      // 大文件可能耗时较长，放宽超时
      timeout: 300_000,
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total))
        }
      },
    })
    .then((r) => r.data)
}

/** 删除文档 */
export const deleteDocument = (filename: string) =>
  apiClient
    .delete(`/api/documents/${encodeURIComponent(filename)}`)
    .then((r) => r.data)

/** 预览文档原始文本（对标 ChatGPT/Claude 文件预览） */
export const previewDocument = (filename: string) =>
  apiClient
    .get<DocumentPreview>(`/api/documents/${encodeURIComponent(filename)}/raw`)
    .then((r) => r.data)

/** 从 URL / sitemap / RSS 导入文档（对标 RAGFlow 多数据源） */
export const importUrl = (req: ImportUrlRequest) =>
  apiClient
    .post<ImportUrlResponse>('/api/documents/import-url', req, {
      timeout: 600_000, // sitemap/rss 批量抓取可能很久
    })
    .then((r) => r.data)

/** 构建索引，返回 {message, stats:{entities, relations}} */
export const buildIndex = () =>
  apiClient
    .post<BuildIndexResponse>('/api/documents/build-index')
    .then((r) => r.data)

/** SSE 流式三元组抽取 */
export function extractTriples(
  filename: string,
  onEvent: (event: ExtractSSEEvent) => void,
  onError?: (err: Error) => void,
): AbortController {
  return streamSSE('/api/documents/extract', { filename }, onEvent, onError)
}

/** 多模型 KG 融合抽取（PocketGraphRAG 独有） */
export const extractMultiModel = (
  filename: string,
  models: string[],
  strategy: 'union' | 'intersect' = 'union',
  minConfidence = 0.6,
) =>
  apiClient
    .post<MultiModelExtractResponse>('/api/documents/extract-multi', {
      filename,
      models,
      strategy,
      min_confidence: minConfidence,
    })
    .then((r) => r.data)

export default apiClient
