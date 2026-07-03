import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * 全局错误边界：捕获子树未处理的运行时错误，避免整页白屏。
 * 生产环境下用户可一键刷新恢复；开发环境下额外显示堆栈便于定位。
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // 生产环境可接入错误上报（Sentry 等）；此处仅控制台输出
    console.error('[ErrorBoundary] 未捕获错误：', error, info.componentStack)
  }

  private handleReload = (): void => {
    this.setState({ hasError: false, error: null })
    // 整页刷新是最可靠的恢复方式
    window.location.reload()
  }

  render(): ReactNode {
    if (!this.state.hasError) return this.props.children

    const isDev = import.meta.env.DEV
    const err = this.state.error

    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-6">
        <div className="w-full max-w-md space-y-4 rounded-lg border border-destructive/30 bg-card p-6 shadow-sm">
          <div className="flex items-start gap-3">
            <AlertCircle className="mt-0.5 h-6 w-6 shrink-0 text-destructive" />
            <div className="space-y-1">
              <h2 className="text-lg font-semibold text-foreground">
                应用遇到问题
              </h2>
              <p className="text-sm text-muted-foreground">
                页面发生了未预期的错误。刷新通常可以恢复；如反复出现，请检查后端服务是否正常运行。
              </p>
            </div>
          </div>

          {isDev && err && (
            <details className="rounded-md bg-muted/50 p-3 text-xs">
              <summary className="cursor-pointer font-medium text-foreground">
                错误详情（仅开发环境）
              </summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all text-destructive">
                {err.name}: {err.message}
                {'\n\n'}
                {err.stack}
              </pre>
            </details>
          )}

          <button
            type="button"
            onClick={this.handleReload}
            className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            <RefreshCw className="h-4 w-4" />
            刷新页面
          </button>
        </div>
      </div>
    )
  }
}
