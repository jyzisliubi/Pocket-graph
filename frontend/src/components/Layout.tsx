import { useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from '@/components/Sidebar'
import { TopBar } from '@/components/TopBar'

/** 路由路径到页面标题的映射 */
const routeTitles: Record<string, string> = {
  '/chat': '智能问答',
  '/kg': '知识图谱',
  '/documents': '文档管理',
  '/analytics': '数据分析',
  '/settings': '系统设置',
}

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()
  const title = routeTitles[location.pathname] ?? 'PocketGraphRAG'

  return (
    <div className="min-h-screen bg-background">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* 主内容区，左侧留出侧边栏宽度（md 以上） */}
      <div className="flex min-h-screen flex-col md:pl-64">
        <TopBar
          title={title}
          onMenuClick={() => setSidebarOpen(true)}
        />
        <main className="flex-1 p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
