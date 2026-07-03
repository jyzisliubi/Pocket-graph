import { useEffect, useState } from 'react'
import { Menu, Moon, Sun, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useTheme } from '@/lib/theme'
import { getHealth } from '@/lib/api'
import { cn } from '@/lib/utils'

type ServiceState = 'online' | 'offline' | 'checking'

interface TopBarProps {
  /** 当前页面标题 */
  title: string
  /** 打开移动端侧边栏 */
  onMenuClick: () => void
}

export function TopBar({ title, onMenuClick }: TopBarProps) {
  const { theme, toggleTheme } = useTheme()
  const [serviceState, setServiceState] = useState<ServiceState>('checking')

  // 轮询健康检查接口（每 30 秒）
  useEffect(() => {
    let active = true

    const check = async () => {
      try {
        const health = await getHealth()
        if (!active) return
        setServiceState(health.status === 'ok' ? 'online' : 'offline')
      } catch {
        if (active) setServiceState('offline')
      }
    }

    check()
    const timer = window.setInterval(check, 30_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const statusConfig = {
    online: { label: '服务正常', dot: 'bg-success', variant: 'success' as const },
    offline: { label: '服务离线', dot: 'bg-destructive', variant: 'destructive' as const },
    checking: { label: '检测中', dot: 'bg-warning animate-pulse', variant: 'warning' as const },
  }

  const status = statusConfig[serviceState]

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b bg-background/80 px-4 backdrop-blur-md md:px-6">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={onMenuClick}
          aria-label="打开菜单"
        >
          <Menu className="h-5 w-5" />
        </Button>
        <h1 className="text-base font-semibold md:text-lg">{title}</h1>
      </div>

      <div className="flex items-center gap-2 md:gap-3">
        {/* 服务状态指示灯 */}
        <Badge variant={status.variant} className="gap-1.5">
          <span className={cn('h-1.5 w-1.5 rounded-full', status.dot)} />
          <span className="hidden sm:inline">{status.label}</span>
          <Activity className="h-3 w-3 sm:hidden" />
        </Badge>

        {/* 主题切换按钮 */}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          aria-label="切换主题"
        >
          {theme === 'dark' ? (
            <Sun className="h-5 w-5" />
          ) : (
            <Moon className="h-5 w-5" />
          )}
        </Button>
      </div>
    </header>
  )
}
