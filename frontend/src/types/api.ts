/**
 * API 响应类型定义
 * 与后端 PocketGraphRAG/api_server.py 的 Pydantic 模型保持一致
 */

/** 检索模式 */
export type SearchMode = 'vector' | 'local' | 'global' | 'mix' | 'kg_only'

/** 问答请求参数 */
export interface QARequest {
  query: string
  search_mode?: SearchMode
  use_multihop?: boolean
  top_k?: number
  use_reranker?: boolean
  vector_weight?: number
  use_hyde?: boolean
  use_query_router?: boolean
  use_self_check?: boolean
}

/** 检索来源（知识片段） */
export interface Source {
  entity: string
  text: string
  score: number
}

/** 知识图谱检索路径信息 */
export interface KGPathInfo {
  search_type: string
  seed_entities: string[]
  expanded_entities: string[]
  matched_relations: string[]
}

/** 检索流水线信息 */
export interface PipelineInfo {
  search_mode: string
  query_rewritten: boolean
  multihop_used: boolean
  kg_path: KGPathInfo
}

/** 问答响应（非流式） */
export interface QAResponse {
  answer: string
  sources: Source[]
  pipeline_info: PipelineInfo
  effective_query: string
}

/** 只检索不生成的响应 */
export interface RetrieveResponse {
  sources: Source[]
  kg_path: KGPathInfo
  query: string
}

/** 图谱统计信息 */
export interface GraphStats {
  total_entities: number
  total_relations: number
  total_edges: number
  avg_degree: number
}

/** 图谱节点（兼容 ECharts 力导向图字段） */
export interface GraphNode {
  id: string
  name: string
  degree: number
  category: number
  symbolSize: number
}

/** 图谱边 */
export interface GraphLink {
  source: string
  target: string
  relation: string
}

/** 子图响应 */
export interface SubgraphResponse {
  nodes: GraphNode[]
  links: GraphLink[]
}

/** 实体搜索结果 */
export interface EntitySearchResult {
  entity: string
  degree: number
}

/** PageRank 结果 */
export interface PagerankResponse {
  entity: string
  score: number
}

/** 社区发现结果 */
export interface CommunityResponse {
  entity: string
  community_id: number
}

/** 最短路径结果 */
export interface ShortestPathResponse {
  path: string[]
  length: number
}

/** LLM 后端状态（脱敏后的 API Key） */
export interface ApiKeyMasked {
  configured: boolean
  masked: string
}

/** LLM 后端详细状态 */
export interface LlmStatusResponse {
  provider: string
  provider_label: string
  has_llm: boolean
  ollama: {
    model: string
    api_base: string
    running: boolean | null
  }
  siliconflow: { model: string; api_key: ApiKeyMasked }
  deepseek: { model: string; api_key: ApiKeyMasked }
  dashscope: { model: string; api_key: ApiKeyMasked }
  openai: {
    model: string
    api_base: string
    api_key: ApiKeyMasked
  }
}

/** 健康检查响应 */
export interface HealthResponse {
  status: 'ok' | 'initializing'
  version: string
  search_mode: string
  rag_ready: boolean
  llm: {
    provider: string
    provider_label: string
    model: string
    has_llm: boolean
    ollama_running: boolean | null
  }
}

/** 文档信息 */
export interface DocumentInfo {
  filename: string
  size: number
  uploaded_at: string
}

/** 上传响应 */
export interface UploadResponse {
  filename: string
  path: string
  size: number
  message: string
}

/** 建索引统计 */
export interface BuildIndexStats {
  entities: number
  relations: number
}

/** 建索引响应 */
export interface BuildIndexResponse {
  message: string
  stats: BuildIndexStats
}

/** 三元组抽取 SSE 阶段 */
export type ExtractPhase = 'extracting' | 'done' | 'error'

/** SSE 三元组抽取事件（{phase, message, triples_count}） */
export interface ExtractSSEEvent {
  phase: ExtractPhase
  message: string
  triples_count: number
}

// ==========================
// 系统设置
// ==========================

/** GET /api/settings 返回的 LLM 配置（API Key 已脱敏） */
export interface SettingsResponse {
  provider: string
  provider_label: string
  has_llm: boolean
  ollama: {
    model: string
    api_base: string
    running: boolean | null
  }
  siliconflow: { model: string; api_key_configured: boolean }
  deepseek: { model: string; api_key_configured: boolean }
  dashscope: { model: string; api_key_configured: boolean }
  openai: {
    model: string
    api_base: string
    api_key_configured: boolean
  }
}

/** POST /api/settings 请求体（api_key 为空时保持原值） */
export interface SaveSettingsRequest {
  provider: string
  model: string
  api_base: string
  api_key?: string
}

/** POST /api/settings 响应 */
export interface SaveSettingsResponse {
  message: string
  provider: string
}

/** GET /api/settings/ollama-models 响应 */
export interface OllamaModelsResponse {
  models: string[]
}

/** 搜索参数默认值（前端持久化到 localStorage） */
export interface SearchDefaults {
  search_mode: SearchMode
  top_k: number
  vector_weight: number
}

// ==========================
// SSE 流式事件类型
// ==========================

/** SSE token 事件（流式生成中的增量片段） */
export interface SSETokenEvent {
  type: 'token'
  chunk: string
  full_answer: string
}

/** SSE sources 事件（检索来源与流水线信息） */
export interface SSESourcesEvent {
  type: 'sources'
  sources: Source[]
  pipeline_info: PipelineInfo
}

/** SSE status 事件（流水线阶段状态） */
export interface SSEStatusEvent {
  type: 'status'
  status: string
}

/** SSE done 事件（流式结束） */
export interface SSEDoneEvent {
  type: 'done'
  answer?: string
}

/** SSE error 事件（流式中途异常） */
export interface SSEErrorEvent {
  type: 'error'
  message: string
}

/** SSE 事件联合类型 */
export type SSEEvent =
  | SSETokenEvent
  | SSESourcesEvent
  | SSEStatusEvent
  | SSEDoneEvent
  | SSEErrorEvent
