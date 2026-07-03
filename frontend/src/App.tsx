import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import ChatPage from '@/pages/Chat'
import KnowledgeGraphPage from '@/pages/KnowledgeGraph'
import DocumentsPage from '@/pages/Documents'
import SettingsPage from '@/pages/Settings'
import AnalyticsPage from '@/pages/Analytics'

export default function App() {
  return (
    <Routes>
      {/* 所有页面共享 Layout（侧边栏 + 顶栏） */}
      <Route element={<Layout />}>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/kg" element={<KnowledgeGraphPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        {/* 默认重定向到问答页 */}
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Route>
    </Routes>
  )
}
