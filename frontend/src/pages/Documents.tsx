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
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import {
  getDocuments,
  uploadDocument,
  deleteDocument,
  extractTriples,
  buildIndex,
} from '@/lib/api'
import type {
  DocumentInfo,
  ExtractPhase,
  BuildIndexStats,
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

  // ===== 删除确认 =====
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

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
            <div className="min-w-[520px] overflow-x-auto rounded-md border">
              <div className="grid grid-cols-[1fr_90px_140px_180px] items-center gap-2 border-b bg-muted/50 px-3 py-2 text-xs font-medium text-muted-foreground">
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
                      className="grid grid-cols-[1fr_90px_140px_180px] items-center gap-2 px-3 py-2.5 text-sm transition-colors hover:bg-accent/40"
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
                          onClick={() => handleExtract(doc.filename)}
                          disabled={extract.open}
                        >
                          <Wand2 className="h-3.5 w-3.5" />
                          <span>抽取</span>
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
