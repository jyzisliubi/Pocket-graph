import { useEffect, useState } from 'react'
import {
  BarChart3,
  Database,
  GitBranch,
  Network,
  TrendingUp,
  Users,
  Activity,
  Cpu,
  RefreshCw,
} from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import {
  getHealth,
  getGraphStats,
  getTopEntities,
  getPagerank,
  getCommunities,
  getDocuments,
} from '@/lib/api'
import type {
  HealthResponse,
  GraphStats,
  EntitySearchResult,
  PagerankResponse,
  CommunityResponse,
  DocumentInfo,
} from '@/types/api'

/** 指标卡片配置 */
interface MetricCard {
  label: string
  value: string | number
  icon: React.ReactNode
  hint?: string
}

/** 单条横向条形图 */
function BarRow({
  label,
  value,
  max,
  suffix,
}: {
  label: string
  value: number
  max: number
  suffix?: string
}) {
  const pct = max > 0 ? Math.max(2, (value / max) * 100) : 0
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="truncate font-medium" title={label}>
          {label}
        </span>
        <span className="ml-2 shrink-0 tabular-nums text-muted-foreground">
          {value.toFixed(4)}
          {suffix}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function AnalyticsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [topEntities, setTopEntities] = useState<EntitySearchResult[]>([])
  const [pagerank, setPagerank] = useState<PagerankResponse[]>([])
  const [communities, setCommunities] = useState<CommunityResponse[]>([])
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadAll = async () => {
    setLoading(true)
    setError(null)
    try {
      const [h, s, e, p, c, d] = await Promise.all([
        getHealth().catch(() => null),
        getGraphStats().catch(() => null),
        getTopEntities(10).catch(() => []),
        getPagerank(10).catch(() => []),
        getCommunities().catch(() => []),
        getDocuments().catch(() => []),
      ])
      setHealth(h)
      setStats(s)
      setTopEntities(e)
      setPagerank(p)
      setCommunities(c)
      setDocuments(d)
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAll()
  }, [])

  // 社区分布聚合
  const communityBuckets = (() => {
    const m = new Map<number, number>()
    for (const c of communities) {
      m.set(c.community_id, (m.get(c.community_id) ?? 0) + 1)
    }
    return Array.from(m.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
  })()

  const maxDegree = topEntities.length
    ? Math.max(...topEntities.map((e) => e.degree))
    : 1
  const maxScore = pagerank.length
    ? Math.max(...pagerank.map((e) => e.score))
    : 1
  const maxCommunityCount = communityBuckets.length
    ? Math.max(...communityBuckets.map(([, n]) => n))
    : 1

  const metrics: MetricCard[] = [
    {
      label: '实体总数',
      value: stats?.total_entities ?? '—',
      icon: <Database className="h-4 w-4" />,
      hint: 'Knowledge Graph Entities',
    },
    {
      label: '关系类型',
      value: stats?.total_relations ?? '—',
      icon: <GitBranch className="h-4 w-4" />,
      hint: 'Distinct Relation Types',
    },
    {
      label: '三元组边数',
      value: stats?.total_edges ?? '—',
      icon: <Network className="h-4 w-4" />,
      hint: 'Total Triple Edges',
    },
    {
      label: '平均度数',
      value: stats?.avg_degree?.toFixed(2) ?? '—',
      icon: <Activity className="h-4 w-4" />,
      hint: 'Average Node Degree',
    },
    {
      label: '社区数量',
      value: communityBuckets.length || '—',
      icon: <Users className="h-4 w-4" />,
      hint: 'Label Propagation Communities',
    },
    {
      label: '文档数',
      value: documents.length,
      icon: <BarChart3 className="h-4 w-4" />,
      hint: 'Uploaded Source Documents',
    },
  ]

  return (
    <div className="space-y-6">
      {/* 顶部标题与刷新 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <TrendingUp className="h-6 w-6" />
            数据分析
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            知识图谱结构、检索系统状态与文档存储概览
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={loadAll}
          disabled={loading}
        >
          <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
          刷新
        </Button>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="py-4 text-sm text-destructive">
            加载失败：{error}
          </CardContent>
        </Card>
      )}

      {/* 指标卡片网格 */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        {loading
          ? Array.from({ length: 6 }).map((_, i) => (
              <Card key={`skeleton-${i}`}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="h-3 w-16 animate-pulse rounded bg-muted" />
                    <div className="h-3 w-3 animate-pulse rounded bg-muted" />
                  </div>
                  <div className="mt-2 h-7 w-20 animate-pulse rounded bg-muted" />
                  <div className="mt-1 h-2 w-12 animate-pulse rounded bg-muted" />
                </CardContent>
              </Card>
            ))
          : metrics.map((m) => (
          <Card key={m.label}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">{m.label}</span>
                <span className="text-muted-foreground">{m.icon}</span>
              </div>
              <div className="mt-2 text-2xl font-bold tabular-nums">
                {m.value}
              </div>
              {m.hint && (
                <div className="mt-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                  {m.hint}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* 系统状态 */}
      {health && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Cpu className="h-4 w-4" />
              系统状态
            </CardTitle>
            <CardDescription>当前 RAG 系统与 LLM 后端运行状态</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <Badge
                variant={health.rag_ready ? 'default' : 'secondary'}
                className="gap-1"
              >
                <span
                  className={cn(
                    'h-1.5 w-1.5 rounded-full',
                    health.rag_ready ? 'bg-background' : 'bg-muted-foreground',
                  )}
                />
                RAG {health.rag_ready ? '就绪' : '初始化中'}
              </Badge>
              <Badge variant="outline">v{health.version}</Badge>
              <Badge variant="outline">检索: {health.search_mode}</Badge>
              {health.llm && (
                <>
                  <Badge variant="outline">
                    LLM: {health.llm.provider_label}
                  </Badge>
                  {health.llm.has_llm && (
                    <Badge variant="outline" className="text-success">
                      LLM 可用
                    </Badge>
                  )}
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 三栏：Top实体 / PageRank / 社区分布 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Top 实体（按度数）</CardTitle>
            <CardDescription>连接最多的核心实体</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {topEntities.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无数据</p>
            ) : (
              topEntities.map((e) => (
                <BarRow
                  key={e.entity}
                  label={e.entity}
                  value={e.degree}
                  max={maxDegree}
                />
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">PageRank Top 10</CardTitle>
            <CardDescription>个性化PageRank重要性分数</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {pagerank.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无数据</p>
            ) : (
              pagerank.map((e) => (
                <BarRow
                  key={e.entity}
                  label={e.entity}
                  value={e.score}
                  max={maxScore}
                />
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">社区分布</CardTitle>
            <CardDescription>标签传播算法聚类（Top 8）</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {communityBuckets.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无数据</p>
            ) : (
              communityBuckets.map(([id, count]) => (
                <BarRow
                  key={id}
                  label={`社区 #${id}`}
                  value={count}
                  max={maxCommunityCount}
                  suffix=" 实体"
                />
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
