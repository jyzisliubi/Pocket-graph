import { NavLink } from 'react-router-dom'
import {
  MessageSquare,
  Network,
  FileText,
  Settings,
  BarChart3,
  X,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'

/** 侧边栏导航项配置 */
const navItems = [
  { to: '/chat', label: '智能问答', icon: MessageSquare },
  { to: '/kg', label: '知识图谱', icon: Network },
  { to: '/documents', label: '文档管理', icon: FileText },
  { to: '/analytics', label: '数据分析', icon: BarChart3 },
  { to: '/settings', label: '系统设置', icon: Settings },
] as const

interface SidebarProps {
  /** 移动端是否展开 */
  open: boolean
  /** 关闭侧边栏（移动端导航后调用） */
  onClose: () => void
}

export function Sidebar({ open, onClose }: SidebarProps) {
  return (
    <>
      {/* 移动端遮罩 */}
      <div
        className={cn(
          'fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition-opacity md:hidden',
          open ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
        onClick={onClose}
      />

      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex h-full w-64 flex-col border-r bg-card transition-transform duration-300 md:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        {/* 品牌 Logo 区 */}
        <div className="flex h-16 items-center justify-between border-b px-6">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Sparkles className="h-5 w-5" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-semibold leading-tight">
                Rice RAG
              </span>
              <span className="text-xs text-muted-foreground leading-tight">
                知识图谱检索
              </span>
            </div>
          </div>
          <button
            className="rounded-md p-1 text-muted-foreground hover:bg-accent md:hidden"
            onClick={onClose}
            aria-label="关闭菜单"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* 导航链接 */}
        <nav className="flex-1 space-y-1 overflow-y-auto p-3">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={onClose}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
                  )
                }
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span>{item.label}</span>
              </NavLink>
            )
          })}
        </nav>

        {/* 底部版本信息 */}
        <div className="border-t p-4">
          <p className="text-xs text-muted-foreground">
            PocketGraphRAG · v0.3.0
          </p>
        </div>
      </aside>
    </>
  )
}
