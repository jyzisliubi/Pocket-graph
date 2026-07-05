import { useEffect, useState, type ComponentType } from 'react'
import {
  Settings as SettingsIcon,
  Cpu,
  Cloud,
  Brain,
  Zap,
  Sparkles,
  RefreshCw,
  Eye,
  EyeOff,
  Check,
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  Save,
  Loader2,
  Download,
  Key,
  ShieldCheck,
} from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import {
  getSettings,
  saveSettings,
  getOllamaModels,
  getHealth,
  getLlmStatus,
} from '@/lib/api'
import type {
  SettingsResponse,
  HealthResponse,
  LlmStatusResponse,
  SearchMode,
  SearchDefaults,
} from '@/types/api'

/** Provider 选项配置 */
interface ProviderOption {
  id: string
  label: string
  icon: ComponentType<{ className?: string }>
  desc: string
  /** 是否需要 API Key */
  needsKey: boolean
  /** 是否可配置 API Base */
  hasApiBase: boolean
}

const PROVIDERS: ProviderOption[] = [
  {
    id: 'ollama',
    label: 'Ollama',
    icon: Cpu,
    desc: '本地运行，无需 API Key',
    needsKey: false,
    hasApiBase: true,
  },
  {
    id: 'siliconflow',
    label: 'SiliconFlow',
    icon: Cloud,
    desc: '硅基流动，提供免费额度',
    needsKey: true,
    hasApiBase: true,
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    icon: Brain,
    desc: '深度求索，性价比高',
    needsKey: true,
    hasApiBase: true,
  },
  {
    id: 'dashscope',
    label: 'DashScope',
    icon: Zap,
    desc: '阿里通义千问',
    needsKey: true,
    hasApiBase: true,
  },
  {
    id: 'openai',
    label: 'OpenAI',
    icon: Sparkles,
    desc: 'GPT 系列',
    needsKey: true,
    hasApiBase: true,
  },
]

/** 各 Provider 的默认配置 */
const PROVIDER_DEFAULTS: Record<
  string,
  { model: string; api_base: string }
> = {
  ollama: { model: '', api_base: 'http://localhost:11434/v1' },
  siliconflow: {
    model: 'qwen-flash',
    api_base: 'https://api.siliconflow.cn/v1',
  },
  deepseek: { model: 'deepseek-chat', api_base: 'https://api.deepseek.com/v1' },
  dashscope: {
    model: 'qwen-plus',
    api_base: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  },
  openai: { model: 'gpt-4o-mini', api_base: 'https://api.openai.com/v1' },
}

/** 搜索模式选项 */
const SEARCH_MODES: { value: SearchMode; label: string }[] = [
  { value: 'vector', label: '向量检索' },
  { value: 'local', label: '本地检索' },
  { value: 'global', label: '全局检索' },
  { value: 'mix', label: '混合检索' },
  { value: 'kg_only', label: '仅知识图谱' },
]

/** 搜索参数 localStorage 键名 */
const SEARCH_DEFAULTS_KEY = 'pocketgraphrag-search-defaults'

/** 搜索参数默认值 */
const SEARCH_DEFAULTS: SearchDefaults = {
  search_mode: 'mix',
  top_k: 5,
  vector_weight: 0.5,
}

/** 单个 Provider 的表单数据 */
interface ProviderForm {
  model: string
  api_base: string
  /** 用户输入的新 API Key，空表示不修改 */
  api_key: string
}

/** 初始化各 Provider 表单 */
function initForms(): Record<string, ProviderForm> {
  const forms: Record<string, ProviderForm> = {}
  for (const p of PROVIDERS) {
    forms[p.id] = {
      ...PROVIDER_DEFAULTS[p.id],
      api_key: '',
    }
  }
  return forms
}

/** 从 localStorage 读取搜索参数默认值 */
function loadSearchDefaults(): SearchDefaults {
  try {
    const raw = localStorage.getItem(SEARCH_DEFAULTS_KEY)
    if (!raw) return { ...SEARCH_DEFAULTS }
    const parsed = JSON.parse(raw) as Partial<SearchDefaults>
    return { ...SEARCH_DEFAULTS, ...parsed }
  } catch {
    return { ...SEARCH_DEFAULTS }
  }
}

/** 提示消息 */
interface StatusMessage {
  type: 'success' | 'error' | 'info'
  text: string
}

/** 密码输入框：带显示/隐藏切换 */
function PasswordInput({
  value,
  onChange,
  placeholder,
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative">
      <Input
        type={show ? 'text' : 'password'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        autoComplete="off"
        className="pr-10"
      />
      <button
        type="button"
        onClick={() => setShow((v) => !v)}
        className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground transition-colors hover:text-foreground"
        aria-label={show ? '隐藏' : '显示'}
      >
        {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
      </button>
    </div>
  )
}

/** 表单字段标签 */
function FieldLabel({
  children,
  hint,
}: {
  children: React.ReactNode
  hint?: string
}) {
  return (
    <div className="mb-1.5 flex items-center justify-between">
      <label className="text-sm font-medium text-foreground">{children}</label>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
  )
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [llmStatus, setLlmStatus] = useState<LlmStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<StatusMessage | null>(null)

  // 当前选中的 Provider
  const [selectedProvider, setSelectedProvider] = useState<string>('ollama')

  // 各 Provider 表单数据
  const [forms, setForms] = useState<Record<string, ProviderForm>>(initForms)

  // API Key 是否已配置（来自后端，脱敏）
  const [keyConfigured, setKeyConfigured] = useState<Record<string, boolean>>(
    {},
  )

  // Ollama 模型列表
  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [ollamaModelsLoading, setOllamaModelsLoading] = useState(false)
  const [ollamaModelsLoaded, setOllamaModelsLoaded] = useState(false)

  // 搜索参数默认值
  const [searchParams, setSearchParams] = useState<SearchDefaults>(
    SEARCH_DEFAULTS,
  )

  /** 用后端返回数据回填表单 */
  const applySettings = (s: SettingsResponse) => {
    setSettings(s)
    setSelectedProvider(s.provider)
    setForms((prev) => ({
      ...prev,
      ollama: {
        model: s.ollama.model || prev.ollama.model,
        api_base: s.ollama.api_base || prev.ollama.api_base,
        api_key: '',
      },
      siliconflow: {
        model: s.siliconflow.model || prev.siliconflow.model,
        api_base: prev.siliconflow.api_base,
        api_key: '',
      },
      deepseek: {
        model: s.deepseek.model || prev.deepseek.model,
        api_base: prev.deepseek.api_base,
        api_key: '',
      },
      dashscope: {
        model: s.dashscope.model || prev.dashscope.model,
        api_base: prev.dashscope.api_base,
        api_key: '',
      },
      openai: {
        model: s.openai.model || prev.openai.model,
        api_base: s.openai.api_base || prev.openai.api_base,
        api_key: '',
      },
    }))
    setKeyConfigured({
      ollama: true,
      siliconflow: s.siliconflow.api_key_configured,
      deepseek: s.deepseek.api_key_configured,
      dashscope: s.dashscope.api_key_configured,
      openai: s.openai.api_key_configured,
    })
  }

  /** 刷新设置与健康状态 */
  const refreshStatus = async () => {
    const [s, h, l] = await Promise.all([
      getSettings(),
      getHealth().catch(() => null),
      getLlmStatus().catch(() => null),
    ])
    applySettings(s)
    setHealth(h)
    setLlmStatus(l)
  }

  /** 拉取 Ollama 模型列表 */
  const refreshOllamaModels = async () => {
    setOllamaModelsLoading(true)
    try {
      const resp = await getOllamaModels()
      setOllamaModels(resp.models || [])
      setOllamaModelsLoaded(true)
    } catch (err) {
      setMessage({
        type: 'error',
        text: `获取模型列表失败：${err instanceof Error ? err.message : '未知错误'}`,
      })
    } finally {
      setOllamaModelsLoading(false)
    }
  }

  // 初始加载
  useEffect(() => {
    let active = true
    ;(async () => {
      setLoading(true)
      try {
        await refreshStatus()
      } catch (err) {
        if (active) {
          setMessage({
            type: 'error',
            text: `加载设置失败：${err instanceof Error ? err.message : '未知错误'}`,
          })
        }
      } finally {
        if (active) setLoading(false)
      }
    })()
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 加载搜索参数默认值
  useEffect(() => {
    setSearchParams(loadSearchDefaults())
  }, [])

  // 选中 Ollama 时自动加载模型列表
  useEffect(() => {
    if (selectedProvider === 'ollama' && !ollamaModelsLoaded) {
      refreshOllamaModels()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProvider])

  /** 更新某个 Provider 的表单字段 */
  const updateForm = (
    provider: string,
    field: keyof ProviderForm,
    value: string,
  ) => {
    setForms((prev) => ({
      ...prev,
      [provider]: { ...prev[provider], [field]: value },
    }))
  }

  /** 保存设置 */
  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const form = forms[selectedProvider]
      const payload = {
        provider: selectedProvider,
        model: form.model.trim(),
        api_base: form.api_base.trim(),
        // API Key 为空时不传，后端保持原值
        ...(form.api_key.trim() ? { api_key: form.api_key.trim() } : {}),
      }
      await saveSettings(payload)

      // 搜索参数持久化到 localStorage
      localStorage.setItem(
        SEARCH_DEFAULTS_KEY,
        JSON.stringify(searchParams),
      )

      setMessage({ type: 'success', text: '设置已保存' })
      // 清空已保存的 API Key 输入
      setForms((prev) => ({
        ...prev,
        [selectedProvider]: { ...prev[selectedProvider], api_key: '' },
      }))
      // 刷新状态
      await refreshStatus()
    } catch (err) {
      setMessage({
        type: 'error',
        text: `保存失败：${err instanceof Error ? err.message : '未知错误'}`,
      })
    } finally {
      setSaving(false)
    }
  }

  /** 拉取模型提示（无对应后端接口，提示用 CLI） */
  const [pullHint, setPullHint] = useState(false)

  /** 当前选中 Provider 的连接状态 */
  const getConnectionStatus = (): {
    label: string
    color: string
  } => {
    const p = selectedProvider
    if (p === 'ollama') {
      if (settings?.ollama.running === true)
        return { label: '运行中', color: 'bg-success' }
      if (settings?.ollama.running === false)
        return { label: '未运行', color: 'bg-destructive' }
      return { label: '未知', color: 'bg-warning' }
    }
    // 云端 Provider：仅当为当前激活 provider 时可判断连接状态
    if (p === settings?.provider) {
      return settings?.has_llm
        ? { label: '已连接', color: 'bg-success' }
        : { label: '连接失败', color: 'bg-destructive' }
    }
    return { label: '未启用', color: 'bg-muted-foreground' }
  }

  const selectedForm = forms[selectedProvider]
  const selectedOption =
    PROVIDERS.find((p) => p.id === selectedProvider) ?? PROVIDERS[0]
  const SelectedIcon = selectedOption.icon
  const conn = getConnectionStatus()
  const hasLlm = health?.llm.has_llm ?? settings?.has_llm ?? false

  // Ollama 下拉选项：合并已加载列表与当前模型，去重
  const ollamaOptions = Array.from(
    new Set(
      [...ollamaModels, forms.ollama.model].filter(Boolean),
    ),
  )

  // 加载中
  if (loading) {
    return (
      <Card className="h-full">
        <CardContent className="flex h-96 items-center justify-center">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            正在加载设置…
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="flex flex-col gap-4 pb-24">
      {/* 标题卡片 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <SettingsIcon className="h-5 w-5" />
            系统设置
          </CardTitle>
          <CardDescription>
            配置 LLM 后端与检索参数。API Key 仅在本地保存到后端，不会明文展示。
          </CardDescription>
        </CardHeader>
      </Card>

      {/* 状态指示区 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">运行状态</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-3">
            {/* 当前 Provider 连接状态 */}
            <div className="flex items-center gap-2 rounded-md border bg-muted/30 px-3 py-2">
              <span
                className={cn('h-2.5 w-2.5 rounded-full', conn.color)}
              />
              <span className="text-sm text-muted-foreground">
                {selectedOption.label}：
              </span>
              <span className="text-sm font-medium">{conn.label}</span>
            </div>

            {/* API Key 配置状态 */}
            {selectedOption.needsKey && (
              <div className="flex items-center gap-2 rounded-md border bg-muted/30 px-3 py-2">
                <Key className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">API Key：</span>
                {keyConfigured[selectedProvider] ? (
                  <Badge variant="success" className="gap-1 px-2 py-0 text-xs">
                    <Check className="h-3 w-3" />
                    已配置
                  </Badge>
                ) : (
                  <Badge variant="warning" className="px-2 py-0 text-xs">
                    未配置
                  </Badge>
                )}
              </div>
            )}

            {/* LLM 是否可用 */}
            <div className="flex items-center gap-2 rounded-md border bg-muted/30 px-3 py-2">
              <span className="text-sm text-muted-foreground">LLM 可用：</span>
              {hasLlm ? (
                <Badge variant="success" className="gap-1 px-2 py-0 text-xs">
                  <Check className="h-3 w-3" />
                  是
                </Badge>
              ) : (
                <Badge variant="destructive" className="px-2 py-0 text-xs">
                  否
                </Badge>
              )}
            </div>

            {/* 激活 Provider 标签 */}
            {settings && (
              <div className="flex items-center gap-2 rounded-md border bg-muted/30 px-3 py-2">
                <span className="text-sm text-muted-foreground">
                  当前激活：
                </span>
                <span className="text-sm font-medium">
                  {settings.provider_label || settings.provider}
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* LLM Provider 选择 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">LLM Provider 选择</CardTitle>
          <CardDescription>选择大语言模型后端，点击卡片切换。</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {PROVIDERS.map((p) => {
              const Icon = p.icon
              const active = p.id === selectedProvider
              const isCurrent = p.id === settings?.provider
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setSelectedProvider(p.id)}
                  className={cn(
                    'relative flex flex-col items-start gap-2 rounded-lg border p-3 text-left transition-colors',
                    active
                      ? 'border-primary bg-accent'
                      : 'border-border bg-card hover:bg-accent/50',
                  )}
                >
                  <div className="flex w-full items-center justify-between">
                    <Icon
                      className={cn(
                        'h-5 w-5',
                        active ? 'text-primary' : 'text-muted-foreground',
                      )}
                    />
                    {active && (
                      <Check className="h-4 w-4 text-primary" />
                    )}
                  </div>
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium">{p.label}</div>
                    <div className="text-xs text-muted-foreground">
                      {p.desc}
                    </div>
                  </div>
                  {isCurrent && (
                    <Badge
                      variant="secondary"
                      className="absolute -top-2 right-2 px-1.5 py-0 text-[10px]"
                    >
                      当前
                    </Badge>
                  )}
                </button>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Provider 配置表单 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <SelectedIcon className="h-4 w-4" />
            {selectedOption.label} 配置
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Ollama：模型下拉 + 刷新/拉取按钮 */}
          {selectedProvider === 'ollama' && (
            <>
              <div>
                <FieldLabel hint="从本地 Ollama 实例获取">模型</FieldLabel>
                <div className="flex items-center gap-2">
                  <Select
                    value={forms.ollama.model || undefined}
                    onValueChange={(v) => updateForm('ollama', 'model', v)}
                  >
                    <SelectTrigger className="flex-1">
                      <SelectValue
                        placeholder={
                          ollamaModelsLoaded
                            ? '选择已安装的模型'
                            : '尚未加载模型列表'
                        }
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {ollamaOptions.length === 0 ? (
                        <SelectItem value="__none__" disabled>
                          <span className="text-muted-foreground">
                            暂无可用模型
                          </span>
                        </SelectItem>
                      ) : (
                        ollamaOptions.map((m) => (
                          <SelectItem key={m} value={m}>
                            {m}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={refreshOllamaModels}
                    disabled={ollamaModelsLoading}
                    aria-label="刷新模型列表"
                    title="刷新模型列表"
                  >
                    <RefreshCw
                      className={cn(
                        'h-4 w-4',
                        ollamaModelsLoading && 'animate-spin',
                      )}
                    />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setPullHint((v) => !v)}
                    aria-label="拉取模型"
                    title="拉取模型"
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
                {ollamaModelsLoaded && ollamaModels.length === 0 && (
                  <p className="mt-1.5 text-xs text-muted-foreground">
                    未检测到已安装模型，请先通过命令行执行{' '}
                    <code className="rounded bg-muted px-1 py-0.5 font-mono">
                      ollama pull qwen2.5:7b
                    </code>{' '}
                    安装。
                  </p>
                )}
                {pullHint && (
                  <p className="mt-1.5 flex items-start gap-1.5 rounded-md border border-warning/40 bg-warning/10 px-2.5 py-1.5 text-xs text-warning-foreground">
                    <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" />
                    <span>
                      拉取模型需在服务端执行{' '}
                      <code className="rounded bg-muted px-1 py-0.5 font-mono">
                        ollama pull &lt;模型名&gt;
                      </code>
                      ，完成后点击刷新按钮即可看到新模型。
                    </span>
                  </p>
                )}
              </div>

              <div>
                <FieldLabel>API Base</FieldLabel>
                <Input
                  value={selectedForm.api_base}
                  onChange={(e) =>
                    updateForm('ollama', 'api_base', e.target.value)
                  }
                  placeholder="http://localhost:11434/v1"
                />
              </div>
            </>
          )}

          {/* SiliconFlow */}
          {selectedProvider === 'siliconflow' && (
            <>
              <div>
                <FieldLabel hint="留空则保持原值">API Key</FieldLabel>
                <PasswordInput
                  value={selectedForm.api_key}
                  onChange={(v) => updateForm('siliconflow', 'api_key', v)}
                  placeholder="sk-..."
                />
                <a
                  href="https://cloud.siliconflow.cn/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1.5 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                >
                  <ExternalLink className="h-3 w-3" />
                  申请免费 API Key
                </a>
              </div>
              <div>
                <FieldLabel>模型</FieldLabel>
                <Input
                  value={selectedForm.model}
                  onChange={(e) =>
                    updateForm('siliconflow', 'model', e.target.value)
                  }
                  placeholder="qwen-flash"
                />
              </div>
              <div>
                <FieldLabel>API Base</FieldLabel>
                <Input
                  value={selectedForm.api_base}
                  onChange={(e) =>
                    updateForm('siliconflow', 'api_base', e.target.value)
                  }
                  placeholder="https://api.siliconflow.cn/v1"
                />
              </div>
            </>
          )}

          {/* DeepSeek */}
          {selectedProvider === 'deepseek' && (
            <>
              <div>
                <FieldLabel hint="留空则保持原值">API Key</FieldLabel>
                <PasswordInput
                  value={selectedForm.api_key}
                  onChange={(v) => updateForm('deepseek', 'api_key', v)}
                  placeholder="sk-..."
                />
              </div>
              <div>
                <FieldLabel>模型</FieldLabel>
                <Input
                  value={selectedForm.model}
                  onChange={(e) =>
                    updateForm('deepseek', 'model', e.target.value)
                  }
                  placeholder="deepseek-chat"
                />
              </div>
              <div>
                <FieldLabel>API Base</FieldLabel>
                <Input
                  value={selectedForm.api_base}
                  onChange={(e) =>
                    updateForm('deepseek', 'api_base', e.target.value)
                  }
                  placeholder="https://api.deepseek.com/v1"
                />
              </div>
            </>
          )}

          {/* DashScope */}
          {selectedProvider === 'dashscope' && (
            <>
              <div>
                <FieldLabel hint="留空则保持原值">API Key</FieldLabel>
                <PasswordInput
                  value={selectedForm.api_key}
                  onChange={(v) => updateForm('dashscope', 'api_key', v)}
                  placeholder="sk-..."
                />
              </div>
              <div>
                <FieldLabel>模型</FieldLabel>
                <Input
                  value={selectedForm.model}
                  onChange={(e) =>
                    updateForm('dashscope', 'model', e.target.value)
                  }
                  placeholder="qwen-plus"
                />
              </div>
              <div>
                <FieldLabel>API Base</FieldLabel>
                <Input
                  value={selectedForm.api_base}
                  onChange={(e) =>
                    updateForm('dashscope', 'api_base', e.target.value)
                  }
                  placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
                />
              </div>
            </>
          )}

          {/* OpenAI */}
          {selectedProvider === 'openai' && (
            <>
              <div>
                <FieldLabel hint="留空则保持原值">API Key</FieldLabel>
                <PasswordInput
                  value={selectedForm.api_key}
                  onChange={(v) => updateForm('openai', 'api_key', v)}
                  placeholder="sk-..."
                />
              </div>
              <div>
                <FieldLabel>模型</FieldLabel>
                <Input
                  value={selectedForm.model}
                  onChange={(e) =>
                    updateForm('openai', 'model', e.target.value)
                  }
                  placeholder="gpt-4o-mini"
                />
              </div>
              <div>
                <FieldLabel>API Base</FieldLabel>
                <Input
                  value={selectedForm.api_base}
                  onChange={(e) =>
                    updateForm('openai', 'api_base', e.target.value)
                  }
                  placeholder="https://api.openai.com/v1"
                />
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* 搜索参数配置 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">搜索参数默认值</CardTitle>
          <CardDescription>
            配置问答检索的默认参数，保存后持久化在本地浏览器。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* 搜索模式 */}
          <div>
            <FieldLabel>搜索模式默认值</FieldLabel>
            <Select
              value={searchParams.search_mode}
              onValueChange={(v) =>
                setSearchParams((p) => ({ ...p, search_mode: v as SearchMode }))
              }
            >
              <SelectTrigger className="w-full sm:w-[240px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SEARCH_MODES.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* top_k */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">top_k 默认值</label>
              <span className="font-mono text-sm">{searchParams.top_k}</span>
            </div>
            <input
              type="range"
              min={1}
              max={100}
              step={1}
              value={searchParams.top_k}
              onChange={(e) =>
                setSearchParams((p) => ({
                  ...p,
                  top_k: Number(e.target.value),
                }))
              }
              className="range-slider w-full"
            />
            <p className="text-xs text-muted-foreground">
              检索返回的知识片段数量（1-100）。
            </p>
          </div>

          {/* vector_weight */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                vector_weight 默认值
              </label>
              <span className="font-mono text-sm">
                {searchParams.vector_weight.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={searchParams.vector_weight}
              onChange={(e) =>
                setSearchParams((p) => ({
                  ...p,
                  vector_weight: Number(e.target.value),
                }))
              }
              className="range-slider w-full"
            />
            <p className="text-xs text-muted-foreground">
              向量检索权重（0-1），值越大越偏向向量相似度。
            </p>
          </div>
        </CardContent>
      </Card>

      {/* 安全 & 角色 LLM（只读状态展示，配置需修改环境变量后重启） */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5" />
            安全 & 角色 LLM
          </CardTitle>
          <CardDescription>
            API 认证与多角色 LLM 配置状态（只读，修改需调整环境变量后重启服务）
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* API 认证 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">API 认证</label>
              <Badge
                variant={llmStatus?.api_auth?.enabled ? 'default' : 'secondary'}
              >
                {llmStatus?.api_auth?.enabled
                  ? `已启用 (${llmStatus.api_auth.key_count} 个 key)`
                  : '未启用（本地开发模式）'}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              启用后所有 <code className="rounded bg-muted px-1">/api/*</code> 端点
              必须带 <code className="rounded bg-muted px-1">X-API-Key</code> 或
              <code className="rounded bg-muted px-1">Authorization: Bearer</code> 头。
              配置：<code className="rounded bg-muted px-1">POCKET_API_KEYS=k1,k2,k3</code>
            </p>
          </div>

          {/* Langfuse Tracing */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Langfuse Tracing</label>
              <Badge
                variant={llmStatus?.langfuse?.enabled ? 'default' : 'secondary'}
              >
                {llmStatus?.langfuse?.enabled ? '已启用' : '未启用'}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              可选 LLM 可观测性平台（对标 LangSmith）。
              配置：<code className="rounded bg-muted px-1">POCKET_LANGFUSE=1</code>
              + public/secret key
            </p>
          </div>

          {/* 角色 LLM 配置 */}
          <div className="space-y-2">
            <label className="text-sm font-medium">角色 LLM（成本优化）</label>
            <p className="text-xs text-muted-foreground">
              为不同角色独立配置 LLM，实现"强模型抽取 + 快模型查询"的成本优化。
              配置格式：<code className="rounded bg-muted px-1">POCKET_&lt;ROLE&gt;_LLM=provider::model</code>
              或 <code className="rounded bg-muted px-1">POCKET_&lt;ROLE&gt;_MODEL=model</code>
            </p>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              {[
                { key: 'extract', label: 'KG 抽取', env: 'POCKET_EXTRACT_LLM' },
                { key: 'query', label: '问答生成', env: 'POCKET_QUERY_LLM' },
                { key: 'keywords', label: '关键词', env: 'POCKET_KEYWORDS_LLM' },
                { key: 'vlm', label: '多模态', env: 'POCKET_VLM_LLM' },
              ].map((role) => {
                const configured = llmStatus?.role_llm?.[role.key as keyof typeof llmStatus.role_llm]
                return (
                  <div
                    key={role.key}
                    className="rounded-md border p-2 text-center"
                  >
                    <div className="text-xs text-muted-foreground">
                      {role.label}
                    </div>
                    <Badge
                      variant={configured ? 'default' : 'secondary'}
                      className="mt-1"
                    >
                      {configured ? '已配置' : '默认'}
                    </Badge>
                    <div className="mt-1 truncate text-[10px] text-muted-foreground">
                      {role.env}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 保存栏（固定底部） */}
      <div className="fixed inset-x-0 bottom-0 z-30 border-t bg-background/95 backdrop-blur md:left-64">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3 md:px-6">
          {/* 提示消息 */}
          <div className="min-h-0 flex-1">
            {message && (
              <div
                className={cn(
                  'flex items-center gap-1.5 text-sm',
                  message.type === 'success' && 'text-success',
                  message.type === 'error' && 'text-destructive',
                  message.type === 'info' && 'text-muted-foreground',
                )}
              >
                {message.type === 'success' ? (
                  <CheckCircle2 className="h-4 w-4 shrink-0" />
                ) : (
                  <AlertCircle className="h-4 w-4 shrink-0" />
                )}
                <span className="truncate">{message.text}</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={refreshStatus}
              disabled={saving || loading}
            >
              <RefreshCw className="h-4 w-4" />
              刷新状态
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Save className="h-4 w-4" />
              )}
              保存设置
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
