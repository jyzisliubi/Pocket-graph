import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import {
  Send,
  Square,
  Bot,
  User,
  ChevronDown,
  Settings2,
  Loader2,
  AlertCircle,
  Search,
  FileText,
  Network,
  Sparkles,
  Trash2,
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { askQuestionStream } from '@/lib/api'
import type { Source, PipelineInfo, SearchMode } from '@/types/api'

/** 检索模式选项 */
const SEARCH_MODES: { value: SearchMode; label: string }[] = [
  { value: 'vector', label: '向量检索' },
  { value: 'local', label: '本地检索' },
  { value: 'global', label: '全局检索' },
  { value: 'mix', label: '混合检索' },
  { value: 'kg_only', label: '仅知识图谱' },
]

/** 推荐问题（空状态展示，对齐默认 movie_kg 演示数据集） */
const RECOMMENDED_QUESTIONS = [
  '无间道是什么类型的电影？',
  '盗梦空间的导演是谁？',
  '肖申克的救赎讲了什么故事？',
  '克里斯托弗·诺兰导演了哪些电影？',
]

/** localStorage 持久化对话历史（仅保留已完成的消息，流式/错误消息不持久化） */
const CHAT_HISTORY_KEY = 'pocketgraphrag_chat_history'
const CHAT_HISTORY_MAX = 50 // 最多保留最近 50 条，避免 localStorage 溢出

function loadChatHistory(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(CHAT_HISTORY_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as ChatMessage[]
    if (!Array.isArray(parsed)) return []
    // 仅恢复已完成、无错误的消息；丢弃 streaming/error 残留
    return parsed
      .filter((m) => !m.streaming && !m.error)
      .slice(-CHAT_HISTORY_MAX)
  } catch {
    return []
  }
}

function saveChatHistory(messages: ChatMessage[]): void {
  try {
    // 只持久化已完成、无错误的消息，避免恢复出半截的流式状态
    const persistable = messages
      .filter((m) => !m.streaming && !m.error)
      .slice(-CHAT_HISTORY_MAX)
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(persistable))
  } catch {
    // localStorage 满或被禁用时静默失败
  }
}

/** 聊天消息结构 */
interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  pipeline_info?: PipelineInfo
  /** 流式过程中当前阶段提示，如 "正在检索知识库…" */
  status?: string
  /** 错误信息 */
  error?: string
  /** 是否仍在生成中 */
  streaming?: boolean
}

/** 生成简单唯一 id */
const genId = () =>
  Math.random().toString(36).slice(2) + Date.now().toString(36)

/** Markdown 渲染器：支持 GFM、数学公式 */
function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {content}
      </Markdown>
    </div>
  )
}

/** 打字指示器（AI 思考中） */
function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 py-1.5">
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </div>
  )
}

/** 来源引用卡片列表（可折叠） */
function SourcesList({ sources }: { sources: Source[] }) {
  const [expanded, setExpanded] = useState(false)
  if (!sources || sources.length === 0) return null

  return (
    <div className="overflow-hidden rounded-md border bg-background/50">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-xs font-medium text-muted-foreground transition-colors hover:bg-accent"
      >
        <ChevronDown
          className={cn(
            'h-3.5 w-3.5 transition-transform',
            !expanded && '-rotate-90',
          )}
        />
        <FileText className="h-3.5 w-3.5" />
        <span>来源引用 ({sources.length})</span>
      </button>
      {expanded && (
        <div className="space-y-1.5 border-t px-2.5 py-2">
          {sources.map((s, i) => (
            <div
              key={i}
              className="rounded-md border bg-muted/30 p-2"
            >
              <div className="mb-1 flex flex-wrap items-center gap-1">
                <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
                  #{i + 1}
                </Badge>
                <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
                  {s.entity}
                </Badge>
                <Badge
                  variant="success"
                  className="px-1.5 py-0 text-[10px] font-mono"
                >
                  {s.score.toFixed(3)}
                </Badge>
              </div>
              <p className="line-clamp-3 text-xs leading-relaxed text-muted-foreground">
                {s.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** Pipeline 信息 badge 行 */
function PipelineBadges({ info }: { info: PipelineInfo }) {
  if (!info) return null
  const kg = info.kg_path
  const responseModeVariant = (
    mode?: string,
  ): 'destructive' | 'secondary' | 'outline' => {
    if (!mode) return 'outline'
    if (mode === 'refused') return 'destructive'
    if (mode.startsWith('retrieval')) return 'secondary'
    return 'outline'
  }
  return (
    <div className="flex flex-wrap items-center gap-1">
      <Badge variant="outline" className="gap-1 px-1.5 py-0 text-[10px]">
        <Search className="h-2.5 w-2.5" />
        {info.search_mode}
      </Badge>
      {info.response_mode && (
        <Badge
          variant={responseModeVariant(info.response_mode)}
          className="px-1.5 py-0 text-[10px]"
        >
          {info.response_mode}
        </Badge>
      )}
      {info.refused && (
        <Badge variant="destructive" className="px-1.5 py-0 text-[10px]">
          已拒绝
        </Badge>
      )}
      {info.fallback_reason && (
        <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
          回退：{info.fallback_reason}
        </Badge>
      )}
      {info.failure_bucket && (
        <Badge variant="destructive" className="px-1.5 py-0 text-[10px]">
          {info.failure_bucket}
        </Badge>
      )}
      {info.llm_error && (
        <Badge variant="destructive" className="px-1.5 py-0 text-[10px]">
          LLM 错误
        </Badge>
      )}
      {info.query_rewritten && (
        <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
          查询改写
        </Badge>
      )}
      {info.multihop_used && (
        <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
          多跳检索
        </Badge>
      )}
      {info.reranker_used && (
        <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
          重排序
        </Badge>
      )}
      {info.hyde_used && (
        <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
          HyDE
        </Badge>
      )}
      {info.query_routed && (
        <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
          查询路由
        </Badge>
      )}
      {info.self_check_used && (
        <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
          自检
        </Badge>
      )}
      {info.question_type && (
        <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
          类型：{info.question_type}
        </Badge>
      )}
      {info.kg_entities_matched !== undefined && info.kg_entities_matched > 0 && (
        <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
          命中实体：{info.kg_entities_matched}
        </Badge>
      )}
      {info.top_k !== undefined && (
        <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
          top_k：{info.top_k}
        </Badge>
      )}
      {info.vector_weight !== undefined && (
        <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
          向量权重：{info.vector_weight.toFixed(2)}
        </Badge>
      )}
      {kg?.search_type && (
        <Badge variant="outline" className="gap-1 px-1.5 py-0 text-[10px]">
          <Network className="h-2.5 w-2.5" />
          {kg.search_type}
        </Badge>
      )}
      {kg?.seed_entities && kg.seed_entities.length > 0 && (
        <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
          种子实体: {kg.seed_entities.length}
        </Badge>
      )}
      {kg?.expanded_entities && kg.expanded_entities.length > 0 && (
        <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
          扩展实体: {kg.expanded_entities.length}
        </Badge>
      )}
      {kg?.matched_relations && kg.matched_relations.length > 0 && (
        <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
          命中关系: {kg.matched_relations.length}
        </Badge>
      )}
    </div>
  )
}

/** 单条消息气泡 */
function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  const showTyping = !isUser && message.streaming && !message.content

  return (
    <div
      className={cn(
        'flex w-full gap-2',
        isUser ? 'justify-end' : 'justify-start',
      )}
    >
      {!isUser && (
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
          <Bot className="h-4 w-4" />
        </div>
      )}

      <div
        className={cn(
          'flex min-w-0 flex-col gap-1.5',
          isUser ? 'max-w-[80%] items-end' : 'max-w-[85%] flex-1 items-start',
        )}
      >
        {/* 气泡主体 */}
        {showTyping ? (
          <div className="rounded-2xl rounded-bl-sm bg-muted px-3 py-1">
            <TypingIndicator />
          </div>
        ) : message.content ? (
          <div
            className={cn(
              'rounded-2xl px-3 py-2',
              isUser
                ? 'rounded-br-sm bg-primary text-primary-foreground'
                : 'rounded-bl-sm bg-muted text-foreground',
            )}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.content}
              </p>
            ) : (
              <>
                <MarkdownRenderer content={message.content} />
                {message.streaming && <span className="stream-cursor" />}
              </>
            )}
          </div>
        ) : null}

        {/* 错误提示 */}
        {message.error && (
          <div className="flex items-start gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{message.error}</span>
          </div>
        )}

        {/* 来源引用 */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <SourcesList sources={message.sources} />
        )}

        {/* Pipeline 信息 */}
        {!isUser && message.pipeline_info && (
          <PipelineBadges info={message.pipeline_info} />
        )}
      </div>

      {isUser && (
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  )
}

/** 空状态欢迎界面 */
function WelcomeScreen({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-5 p-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/15 text-primary">
        <Sparkles className="h-7 w-7" />
      </div>
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">智能问答</h2>
        <p className="mx-auto max-w-md text-sm leading-relaxed text-muted-foreground">
          基于 PocketGraphRAG 知识图谱增强检索系统，向 AI 提问关于电影、
          导演、类型等问题，获取精准答案与来源引用。
        </p>
      </div>
      <div className="grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {RECOMMENDED_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="rounded-lg border bg-card px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadChatHistory())
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [statusText, setStatusText] = useState('')

  // 检索参数
  const [searchMode, setSearchMode] = useState<SearchMode>('mix')
  const [topK, setTopK] = useState(5)
  const [vectorWeight, setVectorWeight] = useState(0.5)
  const [useMultihop, setUseMultihop] = useState(false)
  const [useHyde, setUseHyde] = useState(false)
  const [useReranker, setUseReranker] = useState(false)
  const [useSelfCheck, setUseSelfCheck] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // 流式控制器 & 滚动容器
  const controllerRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const isNearBottomRef = useRef(true)

  // 持久化对话历史到 localStorage（防抖，避免流式过程中频繁写入）
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(() => saveChatHistory(messages), 400)
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    }
  }, [messages])

  /** 清空对话历史 */
  const handleClearChat = () => {
    if (isLoading) return // 生成中不允许清空
    setMessages([])
    try {
      localStorage.removeItem(CHAT_HISTORY_KEY)
    } catch {
      // ignore
    }
  }

  /** 局部更新某条消息 */
  const patchMsg = (id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    )
  }

  /** 发送问题（可传入推荐问题直接发送） */
  const handleSend = (overrideQuery?: string) => {
    const query = (overrideQuery ?? input).trim()
    if (!query || isLoading) return

    const aiMsgId = genId()
    setMessages((prev) => [
      ...prev,
      { id: genId(), role: 'user', content: query },
      {
        id: aiMsgId,
        role: 'assistant',
        content: '',
        status: '正在思考…',
        streaming: true,
      },
    ])
    setInput('')
    setIsLoading(true)
    setError(null)
    setStatusText('正在思考…')

    const controller = askQuestionStream(
      {
        query,
        search_mode: searchMode,
        top_k: topK,
        use_multihop: useMultihop,
        use_hyde: useHyde,
        use_reranker: useReranker,
        use_self_check: useSelfCheck,
        vector_weight: vectorWeight,
      },
      (event) => {
        switch (event.type) {
          case 'status':
            patchMsg(aiMsgId, { status: event.status })
            setStatusText(event.status)
            break
          case 'sources':
            patchMsg(aiMsgId, {
              sources: event.sources,
              pipeline_info: event.pipeline_info,
            })
            break
          case 'token':
            patchMsg(aiMsgId, { content: event.full_answer })
            break
          case 'done': {
            const patch: Partial<ChatMessage> = {
              status: undefined,
              streaming: false,
            }
            if (event.answer !== undefined && event.answer !== '') {
              patch.content = event.answer
            }
            patchMsg(aiMsgId, patch)
            setIsLoading(false)
            setStatusText('')
            controllerRef.current = null
            break
          }
          case 'error':
            patchMsg(aiMsgId, {
              error: event.message,
              status: undefined,
              streaming: false,
            })
            setIsLoading(false)
            setStatusText('')
            setError(event.message)
            controllerRef.current = null
            break
        }
      },
      (err) => {
        patchMsg(aiMsgId, {
          error: err.message,
          status: undefined,
          streaming: false,
        })
        setIsLoading(false)
        setStatusText('')
        setError(err.message)
        controllerRef.current = null
      },
    )
    controllerRef.current = controller
  }

  /** 中断生成 */
  const handleStop = () => {
    controllerRef.current?.abort()
    controllerRef.current = null
    setIsLoading(false)
    setStatusText('')
    setMessages((prev) =>
      prev.map((m) => {
        if (m.role === 'assistant' && m.streaming) {
          return {
            ...m,
            streaming: false,
            status: undefined,
            content: m.content || '（已中断生成）',
          }
        }
        return m
      }),
    )
  }

  // 文本框自动增高
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [input])

  // 消息变化时自动滚动到底部（仅在用户已贴近底部时）
  useEffect(() => {
    const el = scrollRef.current
    if (el && isNearBottomRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages])

  // 组件卸载时中断进行中的请求
  useEffect(() => {
    return () => {
      controllerRef.current?.abort()
    }
  }, [])

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    isNearBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  return (
    <div className="flex h-[calc(100vh-6rem)] flex-col gap-3 md:h-[calc(100vh-7rem)]">
      {/* 顶部工具栏：仅有消息时显示清空按钮 */}
      {messages.length > 0 && !isLoading && (
        <div className="flex justify-end">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClearChat}
            className="h-7 gap-1.5 text-xs text-muted-foreground hover:text-destructive"
            aria-label="清空对话历史"
          >
            <Trash2 className="h-3.5 w-3.5" />
            清空对话
          </Button>
        </div>
      )}

      {/* 消息列表 */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-y-auto rounded-lg border bg-card p-3 md:p-4"
      >
        {messages.length === 0 ? (
          <WelcomeScreen onPick={(q) => handleSend(q)} />
        ) : (
          <div className="flex flex-col gap-4">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
          </div>
        )}
      </div>

      {/* 输入区 */}
      <Card className="shrink-0">
        <CardContent className="space-y-2 p-3">
          {/* 流式状态提示 */}
          {isLoading && statusText && (
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin" />
              <span>{statusText}</span>
            </div>
          )}

          {/* 顶层错误提示 */}
          {error && !isLoading && (
            <div className="flex items-center gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
              <AlertCircle className="h-3.5 w-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* 文本框 + 发送按钮 */}
          <div className="flex items-end gap-2">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              placeholder="请输入你的问题…（Ctrl+Enter 发送）"
              rows={1}
              className="min-h-[44px] max-h-[200px] resize-none"
            />
            {isLoading ? (
              <Button
                variant="destructive"
                size="icon"
                onClick={handleStop}
                aria-label="停止生成"
              >
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                size="icon"
                onClick={() => handleSend()}
                disabled={!input.trim()}
                aria-label="发送"
              >
                <Send className="h-4 w-4" />
              </Button>
            )}
          </div>

          {/* 控制条 */}
          <div className="flex flex-wrap items-center gap-2">
            <Select value={searchMode} onValueChange={(v) => setSearchMode(v as SearchMode)}>
              <SelectTrigger className="h-8 w-[140px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SEARCH_MODES.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    <span className="text-xs">{m.label}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvanced((v) => !v)}
              className="h-8 gap-1.5 px-2 text-xs"
            >
              <Settings2 className="h-3.5 w-3.5" />
              <span>高级选项</span>
              <ChevronDown
                className={cn(
                  'h-3 w-3 transition-transform',
                  showAdvanced && 'rotate-180',
                )}
              />
            </Button>

            <div className="ml-auto text-xs text-muted-foreground">
              {messages.length} 条消息
            </div>
          </div>

          {/* 高级选项面板 */}
          {showAdvanced && (
            <div className="grid grid-cols-1 gap-3 rounded-md border bg-muted/30 p-3 md:grid-cols-2">
              {/* top_k */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-muted-foreground">top_k</label>
                  <span className="font-mono text-xs">{topK}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="range-slider"
                />
              </div>

              {/* vector_weight */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-muted-foreground">
                    vector_weight
                  </label>
                  <span className="font-mono text-xs">
                    {vectorWeight.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={vectorWeight}
                  onChange={(e) => setVectorWeight(Number(e.target.value))}
                  className="range-slider"
                />
              </div>

              {/* 开关组 */}
              <div className="flex items-center justify-between">
                <label className="text-xs">use_multihop</label>
                <Switch checked={useMultihop} onCheckedChange={setUseMultihop} />
              </div>
              <div className="flex items-center justify-between">
                <label className="text-xs">use_hyde</label>
                <Switch checked={useHyde} onCheckedChange={setUseHyde} />
              </div>
              <div className="flex items-center justify-between">
                <label className="text-xs">use_reranker</label>
                <Switch checked={useReranker} onCheckedChange={setUseReranker} />
              </div>
              <div className="flex items-center justify-between">
                <label className="text-xs">use_self_check</label>
                <Switch
                  checked={useSelfCheck}
                  onCheckedChange={setUseSelfCheck}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
