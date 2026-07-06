import { useCallback, useEffect, useRef, useState } from 'react'
import {
  FileText,
  UploadCloud,
  Trash2,
  Wand2,
  Database,
  Loader2,
  AlertCircle,
  CheckCircle2,
  FileCheck2,
  Inbox,
  X,
  Sparkles,
  Eye,
  Globe,
  Rss,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import {
  getDocuments,
  uploadDocument,
  deleteDocument,
  previewDocument,
  importUrl,
  extractTriples,
  extractMultiModel,
  buildIndex,
} from '@/lib/api'
import type {
  DocumentInfo,
  DocumentPreview,
  ImportSourceType,
  ImportUrlResponse,
  ExtractPhase,
  BuildIndexStats,
  MultiModelExtractResponse,
} from '@/types/api'

/** 允许上传的文件扩展名 */
const ALLOWED_EXTS = ['.txt', '.md', '.pdf', '.docx']
/** 文件大小上限：50MB */
const MAX_FILE_SIZE = 50 * 1024 * 1024

/** 把字节数格式化为人类可读的字符串 */
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

/** 格式化 ISO 时间字符串为本地可读时间 */
function formatDate(iso: string): string {
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return iso
    return d.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return iso
  }
}

/** 从文件名提取扩展名（小写，含点） */
function getExt(filename: string): string {
  const idx = filename.lastIndexOf('.')
  return idx >= 0 ? filename.slice(idx).toLowerCase() : ''
}

/** 顶部操作提示（成功 / 错误） */
interface Notice {
  type: 'success' | 'error'
  text: string
}

/** 抽取对话框状态 */
interface ExtractState {
  open: boolean
  filename: string
  phase: ExtractPhase
  message: string
  count: number
}

/** 多模型融合抽取对话框状态 */
interface MultiExtractState {
  open: boolean
  filename: string
  loading: boolean
  error: string | null
  result: MultiModelExtractResponse | null
}

/** 融合策略 */
type FusionStrategy = 'union' | 'intersect'

export default function DocumentsPage() {
  // ===== 文档列表 =====
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [docsError, setDocsError] = useState<string | null>(null)

  // ===== 上传 =====
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadFilename, setUploadFilename] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ===== 抽取 =====
  const [extract, setExtract] = useState<ExtractState>({
    open: false,
    filename: '',
    phase: 'extracting',
    message: '',
    count: 0,
  })
  const extractControllerRef = useRef<AbortController | null>(null)
  const extractDoneTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ===== 多模型融合抽取 =====
  const [multiExtract, setMultiExtract] = useState<MultiExtractState>({
    open: false,
    filename: '',
    loading: false,
    error: null,
    result: null,
  })
  const [fusionModels, setFusionModels] = useState('qwen-flash,qwen-max')
  const [fusionStrategy, setFusionStrategy] =
    useState<FusionStrategy>('union')

  // ===== 删除确认 =====
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  // ===== 文档预览 =====
  const [preview, setPreview] = useState<{
    open: boolean
    loading: boolean
    error: string | null
    data: DocumentPreview | null
  }>({ open: false, loading: false, error: null, data: null })

  // ===== URL / sitemap / RSS 导入 =====
  const [urlImport, setUrlImport] = useState<{
    open: boolean
    loading: boolean
    error: string | null
    result: ImportUrlResponse | null
  }>({ open: false, loading: false, error: null, result: null })
  const [importUrlValue, setImportUrlValue] = useState('')
  const [importType, setImportType] = useState<ImportSourceType>('url')
  const [importMaxItems, setImportMaxItems] = useState(20)

  // ===== 构建索引 =====
  const [building, setBuilding] = useState(false)
  const [indexStats, setIndexStats] = useState<BuildIndexStats | null>(null)
  const [buildError, setBuildError] = useState<string | null>(null)

  // ===== 顶层提示 =====
  const [notice, setNotice] = useState<Notice | null>(null)

  /** 拉取文档列表 */
  const fetchDocuments = useCallback(async () => {
    setLoadingDocs(true)
    setDocsError(null)
    try {
      const list = await getDocuments()
      setDocuments(list ?? [])
    } catch (err) {
      setDocsError((err as Error).message || '获取文档列表失败')
    } finally {
      setLoadingDocs(false)
    }
  }, [])

  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])

  // 组件卸载时中断进行中的 SSE 抽取
  useEffect(() => {
    return () => {
      extractControllerRef.current?.abort()
      if (extractDoneTimerRef.current) clearTimeout(extractDoneTimerRef.current)
    }
  }, [])

  /** 显示一条顶部提示 */
  const showNotice = useCallback((type: 'success' | 'error', text: string) => {
    setNotice({ type, text })
    // 4 秒后自动清除
    setTimeout(() => setNotice(null), 4000)
  }, [])

  /** 校验并上传单个文件 */
  const handleUpload = useCallback(
    async (file: File) => {
      // 校验扩展名
      const ext = getExt(file.name)
      if (!ALLOWED_EXTS.includes(ext)) {
        showNotice(
          'error',
          `不支持的文件格式 ${ext || '（无扩展名）'}，仅支持 ${ALLOWED_EXTS.join(' ')}`,
        )
        return
      }
      // 校验大小
      if (file.size > MAX_FILE_SIZE) {
        showNotice(
          'error',
          `文件过大（${formatSize(file.size)}），最大支持 ${formatSize(MAX_FILE_SIZE)}`,
        )
        return
      }

      setUploading(true)
      setUploadProgress(0)
      setUploadFilename(file.name)
      try {
        await uploadDocument(file, (p) => setUploadProgress(p))
        showNotice('success', `文件 "${file.name}" 上传成功`)
        await fetchDocuments()
      } catch (err) {
        showNotice('error', (err as Error).message || '上传失败')
      } finally {
        setUploading(false)
        setUploadProgress(0)
        setUploadFilename('')
        // 重置 input，便于重复上传同名文件
        if (fileInputRef.current) fileInputRef.current.value = ''
      }
    },
    [fetchDocuments, showNotice],
  )

  /** 处理 input 选择文件 */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
  }

  /** 拖拽放下 */
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleUpload(file)
  }

  /** 触发抽取流程 */
  const handleExtract = (filename: string) => {
    // 重置状态并打开对话框
    setExtract({
      open: true,
      filename,
      phase: 'extracting',
      message: '正在准备抽取…',
      count: 0,
    })

    const controller = extractTriples(
      filename,
      (event) => {
        setExtract((prev) => ({
          ...prev,
          phase: event.phase,
          message: event.message || prev.message,
          count:
            event.triples_count !== undefined
              ? event.triples_count
              : prev.count,
        }))

        if (event.phase === 'done') {
          const total = event.triples_count
          showNotice('success', `成功抽取 ${total} 条三元组`)
          // 1.8 秒后自动关闭对话框，给用户视觉确认
          extractDoneTimerRef.current = setTimeout(() => {
            setExtract((prev) => ({ ...prev, open: false }))
            extractControllerRef.current = null
          }, 1800)
        } else if (event.phase === 'error') {
          showNotice('error', event.message || '抽取失败')
          extractControllerRef.current = null
        }
      },
      (err) => {
        setExtract((prev) => ({
          ...prev,
          phase: 'error',
          message: err.message || '抽取失败',
        }))
        showNotice('error', err.message || '抽取失败')
        extractControllerRef.current = null
      },
    )
    extractControllerRef.current = controller
  }

  /** 关闭抽取对话框（抽取中不可关闭，需先中断） */
  const closeExtract = () => {
    if (extract.phase === 'extracting') {
      // 抽取中关闭视为中断
      extractControllerRef.current?.abort()
      extractControllerRef.current = null
    }
    if (extractDoneTimerRef.current) {
      clearTimeout(extractDoneTimerRef.current)
      extractDoneTimerRef.current = null
    }
    setExtract((prev) => ({ ...prev, open: false }))
  }

  /** 打开多模型融合抽取对话框 */
  const openMultiExtract = (filename: string) => {
    setMultiExtract({
      open: true,
      filename,
      loading: false,
      error: null,
      result: null,
    })
  }

  /** 执行多模型融合抽取 */
  const handleMultiExtract = async () => {
    const models = fusionModels
      .split(',')
      .map((m) => m.trim())
      .filter(Boolean)
    if (models.length < 2) {
      setMultiExtract((prev) => ({
        ...prev,
        error: '请至少输入 2 个模型（逗号分隔）',
      }))
      return
    }
    setMultiExtract((prev) => ({
      ...prev,
      loading: true,
      error: null,
      result: null,
    }))
    try {
      const res = await extractMultiModel(
        multiExtract.filename,
        models,
        fusionStrategy,
      )
      setMultiExtract((prev) => ({
        ...prev,
        loading: false,
        result: res,
      }))
      showNotice(
        'success',
        `融合抽取完成：${res.total_triples} 条三元组（${res.strategy}）`,
      )
    } catch (err) {
      const msg = (err as Error).message || '融合抽取失败'
      setMultiExtract((prev) => ({ ...prev, loading: false, error: msg }))
      showNotice('error', msg)
    }
  }

  /** 关闭多模型融合抽取对话框 */
  const closeMultiExtract = () => {
    if (multiExtract.loading) return // 抽取中不允许关闭
    setMultiExtract((prev) => ({ ...prev, open: false }))
  }

  /** 确认删除 */
  const handleDelete = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await deleteDocument(deleteTarget)
      showNotice('success', `已删除 "${deleteTarget}"`)
      setDeleteTarget(null)
      await fetchDocuments()
    } catch (err) {
      showNotice('error', (err as Error).message || '删除失败')
    } finally {
      setDeleting(false)
    }
  }

  /** 构建索引 */
  const handleBuildIndex = async () => {
    setBuilding(true)
    setBuildError(null)
    try {
      const resp = await buildIndex()
      setIndexStats(resp.stats)
      showNotice('success', resp.message || '索引构建完成')
    } catch (err) {
      setBuildError((err as Error).message || '构建索引失败')
      showNotice('error', (err as Error).message || '构建索引失败')
    } finally {
      setBuilding(false)
    }
  }

  /** 打开文档预览 */
  const handlePreview = async (filename: string) => {
    setPreview({ open: true, loading: true, error: null, data: null })
    try {
      const data = await previewDocument(filename)
      setPreview({ open: true, loading: false, error: null, data })
    } catch (err) {
      setPreview({
        open: true,
        loading: false,
        error: (err as Error).message || '预览加载失败',
        data: null,
      })
    }
  }

  /** 关闭文档预览 */
  const closePreview = () => {
    setPreview((prev) => ({ ...prev, open: false }))
  }

  /** 打开 URL 导入对话框 */
  const openUrlImport = () => {
    setUrlImport({ open: true, loading: false, error: null, result: null })
    setImportUrlValue('')
    setImportType('url')
    setImportMaxItems(20)
  }

  /** 执行 URL 导入 */
  const handleUrlImport = async () => {
    const url = importUrlValue.trim()
    if (!url) {
      setUrlImport((prev) => ({ ...prev, error: '请输入 URL' }))
      return
    }
    if (!/^https?:\/\//i.test(url)) {
      setUrlImport((prev) => ({
        ...prev,
        error: 'URL 必须以 http:// 或 https:// 开头',
      }))
      return
    }
    setUrlImport({ open: true, loading: true, error: null, result: null })
    try {
      const result = await importUrl({
        url,
        source_type: importType,
        max_items: importMaxItems,
      })
      setUrlImport({ open: true, loading: false, error: null, result })
      if (result.imported > 0) {
        showNotice(
          'success',
          `成功导入 ${result.imported} 篇文档（${result.source_type}）`,
        )
        await fetchDocuments()
      } else {
        showNotice('error', result.message || '未导入任何文档')
      }
    } catch (err) {
      const msg = (err as Error).message || 'URL 导入失败'
      setUrlImport({ open: true, loading: false, error: msg, result: null })
      showNotice('error', msg)
    }
  }

  /** 关闭 URL 导入对话框 */
  const closeUrlImport = () => {
    if (urlImport.loading) return // 导入中不允许关闭
    setUrlImport((prev) => ({ ...prev, open: false }))
  }

  return (
    <div className="flex h-[calc(100vh-6rem)] flex-col gap-4 md:h-[calc(100vh-7rem)]">
      {/* 顶部提示 */}
      {notice && (
        <div
          className={cn(
            'flex items-center gap-2 rounded-md border px-3 py-2 text-sm',
            notice.type === 'success'
              ? 'border-success/40 bg-success/10 text-success'
              : 'border-destructive/40 bg-destructive/10 text-destructive',
          )}
        >
          {notice.type === 'success' ? (
            <CheckCircle2 className="h-4 w-4 shrink-0" />
          ) : (
            <AlertCircle className="h-4 w-4 shrink-0" />
          )}
          <span className="flex-1">{notice.text}</span>
          <button
            type="button"
            onClick={() => setNotice(null)}
            className="opacity-70 hover:opacity-100"
            aria-label="关闭提示"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* 上传区 */}
      <Card className="shrink-0">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <UploadCloud className="h-4 w-4" />
            上传文档
            <Button
              variant="outline"
              size="sm"
              className="ml-auto h-7 gap-1.5 px-2 text-xs"
              onClick={openUrlImport}
              disabled={urlImport.loading}
              title="从 URL / sitemap / RSS 批量导入"
            >
              <Globe className="h-3.5 w-3.5" />
              <span>导入 URL</span>
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div
            role="button"
            tabIndex={0}
            onClick={() => !uploading && fileInputRef.current?.click()}
            onKeyDown={(e) => {
              if ((e.key === 'Enter' || e.key === ' ') && !uploading) {
                e.preventDefault()
                fileInputRef.current?.click()
              }
            }}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={cn(
              'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center transition-colors',
              dragOver
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border text-muted-foreground hover:border-primary/60 hover:bg-accent/40',
              uploading && 'pointer-events-none opacity-60',
            )}
          >
            <UploadCloud
              className={cn(
                'h-10 w-10 transition-colors',
                dragOver ? 'text-primary' : 'text-muted-foreground',
              )}
            />
            <div className="space-y-0.5">
              <p className="text-sm font-medium text-foreground">
                {dragOver ? '松开以上传文件' : '拖拽文件到此处，或点击选择文件'}
              </p>
              <p className="text-xs text-muted-foreground">
                支持 {ALLOWED_EXTS.join(' / ')}，最大 {formatSize(MAX_FILE_SIZE)}
              </p>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept={ALLOWED_EXTS.join(',')}
            onChange={handleFileChange}
            className="hidden"
          />

          {/* 上传进度条 */}
          {uploading && (
            <div className="mt-3 space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span className="max-w-[60%] truncate">{uploadFilename}</span>
                </span>
                <span className="font-mono text-primary">{uploadProgress}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-200"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 文档列表 */}
      <Card className="flex min-h-0 flex-1 flex-col">
        <CardHeader className="shrink-0 pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <FileText className="h-4 w-4" />
            文档列表
            <Badge variant="secondary" className="ml-1 px-1.5 py-0 text-[10px]">
              {documents.length}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 overflow-auto pt-0">
          {/* 加载中 */}
          {loadingDocs ? (
            <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              正在加载文档列表…
            </div>
          ) : docsError ? (
            <div className="flex h-40 items-center justify-center rounded-md border border-destructive/40 bg-destructive/10 px-3 text-center text-sm text-destructive">
              <AlertCircle className="mr-2 h-4 w-4 shrink-0" />
              {docsError}
            </div>
          ) : documents.length === 0 ? (
            /* 空状态 */
            <div className="flex h-40 flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
              <Inbox className="h-8 w-8" />
              <span>暂无文档，请先上传文件</span>
            </div>
          ) : (
            /* 列表表头 */
            <div className="min-w-[600px] overflow-x-auto rounded-md border">
              <div className="grid grid-cols-[1fr_90px_140px_240px] items-center gap-2 border-b bg-muted/50 px-3 py-2 text-xs font-medium text-muted-foreground">
                <span>文件名</span>
                <span className="text-right">大小</span>
                <span>上传时间</span>
                <span className="text-right">操作</span>
              </div>
              <div className="divide-y">
                {documents.map((doc) => {
                  const ext = getExt(doc.filename)
                  return (
                    <div
                      key={doc.filename}
                      className="grid grid-cols-[1fr_90px_140px_240px] items-center gap-2 px-3 py-2.5 text-sm transition-colors hover:bg-accent/40"
                    >
                      {/* 文件名 */}
                      <div className="flex min-w-0 items-center gap-2">
                        <FileCheck2 className="h-4 w-4 shrink-0 text-muted-foreground" />
                        <span className="truncate" title={doc.filename}>
                          {doc.filename}
                        </span>
                        {ext && (
                          <Badge
                            variant="outline"
                            className="shrink-0 px-1.5 py-0 text-[10px] uppercase"
                          >
                            {ext.slice(1)}
                          </Badge>
                        )}
                      </div>
                      {/* 大小 */}
                      <span className="text-right font-mono text-xs text-muted-foreground">
                        {formatSize(doc.size)}
                      </span>
                      {/* 上传时间 */}
                      <span className="text-xs text-muted-foreground">
                        {formatDate(doc.uploaded_at)}
                      </span>
                      {/* 操作 */}
                      <div className="flex items-center justify-end gap-1.5">
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 gap-1.5 px-2 text-xs"
                          onClick={() => handlePreview(doc.filename)}
                          disabled={preview.loading}
                          title="预览文档内容"
                        >
                          <Eye className="h-3.5 w-3.5" />
                          <span>预览</span>
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 gap-1.5 px-2 text-xs"
                          onClick={() => handleExtract(doc.filename)}
                          disabled={extract.open || multiExtract.loading}
                        >
                          <Wand2 className="h-3.5 w-3.5" />
                          <span>抽取</span>
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 gap-1.5 px-2 text-xs text-primary hover:bg-primary hover:text-primary-foreground"
                          onClick={() => openMultiExtract(doc.filename)}
                          disabled={extract.open || multiExtract.loading}
                          title="多模型 KG 融合抽取（PocketGraphRAG 独有）"
                        >
                          <Sparkles className="h-3.5 w-3.5" />
                          <span>融合抽取</span>
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 gap-1.5 px-2 text-xs text-destructive hover:bg-destructive hover:text-destructive-foreground"
                          onClick={() => setDeleteTarget(doc.filename)}
                          disabled={deleting}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                          <span>删除</span>
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 底部：构建索引 */}
      <Card className="shrink-0">
        <CardContent className="flex flex-wrap items-center gap-3 p-4">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">索引构建</span>
          </div>

          <Button
            onClick={handleBuildIndex}
            disabled={building}
            className="gap-1.5"
          >
            {building ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Database className="h-4 w-4" />
            )}
            <span>{building ? '构建中…' : '构建索引'}</span>
          </Button>

          {/* 构建错误 */}
          {buildError && !building && (
            <span className="flex items-center gap-1 text-xs text-destructive">
              <AlertCircle className="h-3.5 w-3.5" />
              {buildError}
            </span>
          )}

          {/* 统计结果 */}
          {indexStats && !building && (
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="success" className="gap-1">
                <CheckCircle2 className="h-3 w-3" />
                实体 {indexStats.entities}
              </Badge>
              <Badge variant="secondary" className="gap-1">
                实体关系 {indexStats.relations}
              </Badge>
            </div>
          )}

          <span className="ml-auto text-xs text-muted-foreground">
            抽取三元组后，构建索引以供检索
          </span>
        </CardContent>
      </Card>

      {/* 抽取进度对话框 */}
      <Dialog
        open={extract.open}
        onOpenChange={(open) => {
          if (!open) closeExtract()
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Wand2 className="h-4 w-4" />
              三元组抽取
            </DialogTitle>
            <DialogDescription className="truncate" title={extract.filename}>
              文件：{extract.filename}
            </DialogDescription>
          </DialogHeader>

          {/* 主体内容根据阶段切换 */}
          <div className="space-y-4">
            {extract.phase === 'extracting' && (
              <>
                <div className="flex items-center gap-2 text-sm">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  <span className="text-muted-foreground">
                    {extract.message || '正在抽取三元组…'}
                  </span>
                </div>
                {/* 不确定进度条 */}
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div className="progress-indeterminate-bar h-full w-1/4 rounded-full bg-primary" />
                </div>
                <div className="flex items-center justify-between rounded-md border bg-muted/30 px-3 py-2">
                  <span className="text-xs text-muted-foreground">
                    已抽取三元组
                  </span>
                  <span className="font-mono text-sm font-semibold text-primary">
                    {extract.count} 条
                  </span>
                </div>
              </>
            )}

            {extract.phase === 'done' && (
              <div className="flex flex-col items-center gap-3 py-2 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/15 text-success">
                  <CheckCircle2 className="h-6 w-6" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">抽取完成</p>
                  <p className="text-xs text-muted-foreground">
                    成功抽取{' '}
                    <span className="font-mono font-semibold text-success">
                      {extract.count}
                    </span>{' '}
                    条三元组
                  </p>
                </div>
              </div>
            )}

            {extract.phase === 'error' && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{extract.message || '抽取失败'}</span>
              </div>
            )}
          </div>

          <DialogFooter>
            {extract.phase === 'extracting' ? (
              <Button variant="destructive" onClick={closeExtract}>
                中断抽取
              </Button>
            ) : (
              <Button onClick={closeExtract}>完成</Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 多模型融合抽取对话框 */}
      <Dialog
        open={multiExtract.open}
        onOpenChange={(open) => {
          if (!open) closeMultiExtract()
        }}
      >
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              多模型 KG 融合抽取
              <Badge
                variant="outline"
                className="ml-1 px-1.5 py-0 text-[10px] uppercase text-primary"
              >
                独有
              </Badge>
            </DialogTitle>
            <DialogDescription className="truncate" title={multiExtract.filename}>
              文件：{multiExtract.filename}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* 配置区 */}
            {!multiExtract.result && (
              <>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">
                    模型列表（逗号分隔，至少 2 个）
                  </label>
                  <Input
                    value={fusionModels}
                    onChange={(e) => setFusionModels(e.target.value)}
                    placeholder="qwen-flash,qwen-max"
                    disabled={multiExtract.loading}
                  />
                  <p className="text-[11px] text-muted-foreground">
                    每个模型有盲点，多模型 union 能覆盖彼此遗漏的实体。
                    实测 HotpotQA Hit Rate 0.80 → 0.86（+6%）。
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">
                    融合策略
                  </label>
                  <Select
                    value={fusionStrategy}
                    onValueChange={(v) =>
                      setFusionStrategy(v as FusionStrategy)
                    }
                    disabled={multiExtract.loading}
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="union">
                        union — 并集去重（推荐，高召回）
                      </SelectItem>
                      <SelectItem value="intersect">
                        intersect — 交集（高精度）
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </>
            )}

            {/* 加载中 */}
            {multiExtract.loading && (
              <div className="flex items-center gap-2 text-sm">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                <span className="text-muted-foreground">
                  正在用多个模型抽取并融合…（可能需要 1-2 分钟）
                </span>
              </div>
            )}

            {/* 错误 */}
            {multiExtract.error && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{multiExtract.error}</span>
              </div>
            )}

            {/* 结果 */}
            {multiExtract.result && (
              <div className="space-y-3">
                <div className="flex flex-col items-center gap-2 py-2 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/15 text-success">
                    <CheckCircle2 className="h-6 w-6" />
                  </div>
                  <p className="text-sm font-medium">融合抽取完成</p>
                  <p className="text-xs text-muted-foreground">
                    共{' '}
                    <span className="font-mono font-semibold text-success">
                      {multiExtract.result.total_triples}
                    </span>{' '}
                    条三元组（策略：{multiExtract.result.strategy}）
                  </p>
                </div>
                {/* 各模型抽取数量 */}
                <div className="rounded-md border bg-muted/30 p-3">
                  <p className="mb-2 text-xs font-medium text-muted-foreground">
                    各模型抽取数量
                  </p>
                  <div className="space-y-1">
                    {Object.entries(
                      multiExtract.result.model_stats || {},
                    ).map(([m, n]) => (
                      <div
                        key={m}
                        className="flex items-center justify-between text-xs"
                      >
                        <span className="font-mono">{m}</span>
                        <span className="font-mono font-semibold text-primary">
                          {n} 条
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <DialogFooter>
            {multiExtract.loading ? (
              <Button disabled>抽取中…</Button>
            ) : multiExtract.result ? (
              <Button onClick={closeMultiExtract}>完成</Button>
            ) : (
              <>
                <Button variant="outline" onClick={closeMultiExtract}>
                  取消
                </Button>
                <Button onClick={handleMultiExtract}>
                  <Sparkles className="mr-1 h-3.5 w-3.5" />
                  开始融合抽取
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 文档预览对话框 */}
      <Dialog
        open={preview.open}
        onOpenChange={(open) => {
          if (!open) closePreview()
        }}
      >
        <DialogContent className="max-h-[85vh] max-w-3xl overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Eye className="h-4 w-4" />
              文档预览
              {preview.data && (
                <Badge
                  variant="outline"
                  className="ml-1 px-1.5 py-0 text-[10px] uppercase"
                >
                  {preview.data.source_type}
                </Badge>
              )}
            </DialogTitle>
            <DialogDescription className="truncate">
              {preview.data?.filename || '加载中…'}
            </DialogDescription>
          </DialogHeader>

          {/* 主体内容 */}
          <div className="min-h-0 flex-1 overflow-hidden">
            {preview.loading ? (
              <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                正在提取文档内容…
              </div>
            ) : preview.error ? (
              <div className="flex h-40 items-center justify-center rounded-md border border-destructive/40 bg-destructive/10 px-3 text-center text-sm text-destructive">
                <AlertCircle className="mr-2 h-4 w-4 shrink-0" />
                {preview.error}
              </div>
            ) : preview.data ? (
              <>
                {/* 元信息条 */}
                <div className="mb-2 flex flex-wrap items-center gap-2 border-b pb-2 text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">
                    {preview.data.title}
                  </span>
                  <span>·</span>
                  <span className="font-mono">
                    {preview.data.total_chars.toLocaleString()} 字符
                  </span>
                  {preview.data.truncated && (
                    <Badge
                      variant="outline"
                      className="px-1.5 py-0 text-[10px] text-amber-600 border-amber-400/50"
                    >
                      已截断（仅显示前 50,000 字符）
                    </Badge>
                  )}
                </div>
                {/* 内容区 */}
                <pre className="max-h-[55vh] min-h-[200px] overflow-auto whitespace-pre-wrap break-words rounded-md border bg-muted/30 p-3 text-xs leading-relaxed">
                  {preview.data.content || '（文档内容为空）'}
                </pre>
              </>
            ) : null}
          </div>

          <DialogFooter>
            <Button onClick={closePreview}>关闭</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* URL 导入对话框 */}
      <Dialog
        open={urlImport.open}
        onOpenChange={(open) => {
          if (!open) closeUrlImport()
        }}
      >
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Globe className="h-4 w-4" />
              从 URL 导入文档
              <Badge
                variant="outline"
                className="ml-1 px-1.5 py-0 text-[10px] uppercase text-primary"
              >
                多数据源
              </Badge>
            </DialogTitle>
            <DialogDescription>
              支持 sitemap.xml / RSS feed / 单个网页，批量导入为本地 .txt 文件
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* 配置区 */}
            {!urlImport.result && (
              <>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">
                    数据源 URL
                  </label>
                  <Input
                    value={importUrlValue}
                    onChange={(e) => setImportUrlValue(e.target.value)}
                    placeholder="https://example.com/sitemap.xml"
                    disabled={urlImport.loading}
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-muted-foreground">
                      数据源类型
                    </label>
                    <Select
                      value={importType}
                      onValueChange={(v) =>
                        setImportType(v as ImportSourceType)
                      }
                      disabled={urlImport.loading}
                    >
                      <SelectTrigger className="h-9">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="url">
                          url — 单个网页
                        </SelectItem>
                        <SelectItem value="sitemap">
                          sitemap — 站点地图（批量）
                        </SelectItem>
                        <SelectItem value="rss">
                          rss — RSS/Atom feed（批量）
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-muted-foreground">
                      最大条目数
                    </label>
                    <Input
                      type="number"
                      min={1}
                      max={500}
                      value={importMaxItems}
                      onChange={(e) =>
                        setImportMaxItems(
                          Math.max(1, Math.min(500, Number(e.target.value) || 20)),
                        )
                      }
                      disabled={urlImport.loading}
                    />
                  </div>
                </div>
                <p className="text-[11px] text-muted-foreground">
                  {importType === 'sitemap' && (
                    <span className="flex items-center gap-1">
                      <Globe className="h-3 w-3" />
                      解析 sitemap.xml，逐个抓取页面文本。适合博客/文档站全量导入。
                    </span>
                  )}
                  {importType === 'rss' && (
                    <span className="flex items-center gap-1">
                      <Rss className="h-3 w-3" />
                      解析 RSS/Atom feed，优先用 feed 自带内容。适合新闻/博客订阅。
                    </span>
                  )}
                  {importType === 'url' && (
                    <span className="flex items-center gap-1">
                      <Globe className="h-3 w-3" />
                      抓取单个网页，提取正文（Playwright 渲染失败时回退到 requests）。
                    </span>
                  )}
                </p>
              </>
            )}

            {/* 加载中 */}
            {urlImport.loading && (
              <div className="flex items-center gap-2 text-sm">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                <span className="text-muted-foreground">
                  正在抓取并导入…（sitemap/rss 可能需要几分钟）
                </span>
              </div>
            )}

            {/* 错误 */}
            {urlImport.error && (
              <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{urlImport.error}</span>
              </div>
            )}

            {/* 结果 */}
            {urlImport.result && (
              <div className="space-y-3">
                <div className="flex flex-col items-center gap-2 py-2 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/15 text-success">
                    <CheckCircle2 className="h-6 w-6" />
                  </div>
                  <p className="text-sm font-medium">导入完成</p>
                  <p className="text-xs text-muted-foreground">
                    共导入{' '}
                    <span className="font-mono font-semibold text-success">
                      {urlImport.result.imported}
                    </span>{' '}
                    篇文档（来源：{urlImport.result.source_type}）
                  </p>
                </div>
                {urlImport.result.filenames.length > 0 && (
                  <div className="max-h-40 overflow-auto rounded-md border bg-muted/30 p-3">
                    <p className="mb-2 text-xs font-medium text-muted-foreground">
                      生成的文件
                    </p>
                    <div className="space-y-1">
                      {urlImport.result.filenames.map((f) => (
                        <div
                          key={f}
                          className="flex items-center gap-1.5 text-xs"
                        >
                          <FileCheck2 className="h-3 w-3 shrink-0 text-muted-foreground" />
                          <span className="truncate font-mono" title={f}>
                            {f}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <DialogFooter>
            {urlImport.loading ? (
              <Button disabled>导入中…</Button>
            ) : urlImport.result ? (
              <Button onClick={closeUrlImport}>完成</Button>
            ) : (
              <>
                <Button variant="outline" onClick={closeUrlImport}>
                  取消
                </Button>
                <Button onClick={handleUrlImport}>
                  <Globe className="mr-1 h-3.5 w-3.5" />
                  开始导入
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 删除确认对话框 */}
      <Dialog
        open={!!deleteTarget}
        onOpenChange={(open) => {
          if (!open && !deleting) setDeleteTarget(null)
        }}
      >
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Trash2 className="h-4 w-4 text-destructive" />
              确认删除
            </DialogTitle>
            <DialogDescription>
              确定要删除文档{' '}
              <span className="font-medium text-foreground" title={deleteTarget ?? ''}>
                {deleteTarget}
              </span>{' '}
              吗？此操作不可撤销。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-2">
            <Button
              variant="outline"
              onClick={() => setDeleteTarget(null)}
              disabled={deleting}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleting}
            >
              {deleting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
              确认删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
