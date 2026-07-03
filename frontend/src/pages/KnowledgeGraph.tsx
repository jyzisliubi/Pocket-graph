import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertCircle,
  Download,
  Loader2,
  Maximize2,
  Network,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
  Search,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useTheme } from '@/lib/theme'
import {
  getCommunities,
  getEntitySubgraph,
  getGraphStats,
  getMultiEntitySubgraph,
  getPagerank,
  getShortestPath,
  getTopEntities,
  searchEntities,
} from '@/lib/api'
import type {
  EntitySearchResult,
  GraphStats,
  ShortestPathResponse,
} from '@/types/api'
import {
  ForceGraph,
  type ForceGraphHandle,
  type KgLink,
  type KgNode,
} from '@/components/kg/ForceGraph'
import { GraphSidePanel } from '@/components/kg/GraphSidePanel'

/** 由无向两端生成边 key（与 ForceGraph 内部一致） */
function linkKeyOf(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`
}

/** 限制图谱规模：按度数降序保留前 maxNodes 个节点，并过滤两端均在保留集合中的边 */
function limitGraph(
  nodes: KgNode[],
  links: KgLink[],
  maxNodes: number,
): { nodes: KgNode[]; links: KgLink[] } {
  if (nodes.length <= maxNodes) return { nodes, links }
  const sorted = [...nodes].sort((a, b) => b.degree - a.degree)
  const kept = sorted.slice(0, maxNodes)
  const keptIds = new Set(kept.map((n) => n.id))
  const keptLinks = links.filter(
    (l) => keptIds.has(l.source) && keptIds.has(l.target),
  )
  return { nodes: kept, links: keptLinks }
}

/** 统计卡片 */
function StatCard({
  label,
  value,
  hint,
}: {
  label: string
  value: string | number
  hint?: string
}) {
  return (
    <Card className="py-3">
      <CardContent className="p-0 px-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
        {hint && <p className="mt-0.5 text-[11px] text-muted-foreground">{hint}</p>}
      </CardContent>
    </Card>
  )
}

export default function KnowledgeGraphPage() {
  const { theme } = useTheme()
  const graphRef = useRef<ForceGraphHandle>(null)

  // 图谱数据
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [nodes, setNodes] = useState<KgNode[]>([])
  const [links, setLinks] = useState<KgLink[]>([])
  const [pagerankMap, setPagerankMap] = useState<Map<string, number>>(new Map())
  const [communityMap, setCommunityMap] = useState<Map<string, number>>(
    new Map(),
  )

  // 加载状态
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  // 同步 loading 到 ref，供稳定回调读取最新值以区分首次加载/刷新
  const loadingRef = useRef(true)
  loadingRef.current = loading

  // 交互状态
  const [selectedNode, setSelectedNode] = useState<KgNode | null>(null)
  const [activeTab, setActiveTab] = useState('details')
  const [panelCollapsed, setPanelCollapsed] = useState(false)

  // 搜索状态
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<EntitySearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)

  // 最短路径状态
  const [pathStart, setPathStart] = useState('')
  const [pathEnd, setPathEnd] = useState('')
  const [pathResult, setPathResult] = useState<ShortestPathResponse | null>(
    null,
  )
  const [pathError, setPathError] = useState<string | null>(null)
  const [findingPath, setFindingPath] = useState(false)

  // ==========================
  // 初始加载
  // ==========================
  const loadInitial = useCallback(async () => {
    const isRefresh = !loadingRef.current
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    setLoadError(null)
    setSelectedNode(null)
    setPathResult(null)

    try {
      // 并发拉取统计、实体列表、PageRank、社区
      const [st, topEntities, prList, commList] = await Promise.all([
        getGraphStats(),
        getTopEntities(50),
        getPagerank(50),
        getCommunities(),
      ])
      setStats(st)
      const prMap = new Map(prList.map((r) => [r.entity, r.score]))
      const commMap = new Map(commList.map((r) => [r.entity, r.community_id]))
      setPagerankMap(prMap)
      setCommunityMap(commMap)

      // 取 PageRank + 高度数实体作为种子，调用多实体子图接口获取边结构
      const seedSet = new Set<string>()
      for (const r of prList.slice(0, 25)) seedSet.add(r.entity)
      for (const e of topEntities.slice(0, 25)) seedSet.add(e.entity)
      const seeds = Array.from(seedSet).slice(0, 30)

      let initialNodes: KgNode[] = []
      let initialLinks: KgLink[] = []
      if (seeds.length > 0) {
        const sub = await getMultiEntitySubgraph(seeds, 1)
        initialNodes = sub.nodes.map((n) => ({
          ...n,
          pagerank: prMap.get(n.name),
          communityId: commMap.get(n.name),
        }))
        initialLinks = sub.links
      }

      // 限制规模以保证渲染性能
      const { nodes: limNodes, links: limLinks } = limitGraph(
        initialNodes,
        initialLinks,
        120,
      )
      setNodes(limNodes)
      setLinks(limLinks)
    } catch (e) {
      setLoadError((e as Error).message || '加载图谱失败')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    loadInitial()
  }, [loadInitial])

  // ==========================
  // 合并子图到当前图谱（节点点击/聚焦时扩展邻域）
  // ==========================
  const mergeSubgraph = useCallback(
    (subNodes: KgNode[], subLinks: KgLink[]) => {
      setNodes((prev) => {
        const map = new Map(prev.map((n) => [n.id, n]))
        let changed = false
        for (const n of subNodes) {
          if (!map.has(n.id)) {
            map.set(n.id, {
              ...n,
              pagerank: pagerankMap.get(n.name),
              communityId: communityMap.get(n.name),
            })
            changed = true
          }
        }
        return changed ? Array.from(map.values()) : prev
      })
      setLinks((prev) => {
        const existing = new Set(
          prev.map((l) => `${l.source}|${l.target}|${l.relation}`),
        )
        let changed = false
        const next = [...prev]
        for (const l of subLinks) {
          const k = `${l.source}|${l.target}|${l.relation}`
          if (!existing.has(k)) {
            existing.add(k)
            next.push(l)
            changed = true
          }
        }
        return changed ? next : prev
      })
    },
    [pagerankMap, communityMap],
  )

  // ==========================
  // 节点点击：选中 + 拉取邻域子图
  // ==========================
  const handleNodeClick = useCallback(
    async (node: KgNode) => {
      setSelectedNode(node)
      setPathResult(null)
      setActiveTab('details')
      try {
        const sub = await getEntitySubgraph(node.name, 2)
        mergeSubgraph(sub.nodes as KgNode[], sub.links as KgLink[])
      } catch {
        // 拉取邻域失败不影响选中
      }
    },
    [mergeSubgraph],
  )

  // ==========================
  // 聚焦实体（搜索/路径列表点击）
  // ==========================
  const handleFocusEntity = useCallback(
    async (name: string) => {
      const existing = nodes.find((n) => n.name === name || n.id === name)
      if (existing) {
        setSelectedNode(existing)
        setPathResult(null)
        setActiveTab('details')
        // 同时拉取邻域以高亮邻居
        try {
          const sub = await getEntitySubgraph(name, 2)
          mergeSubgraph(sub.nodes as KgNode[], sub.links as KgLink[])
        } catch {
          // ignore
        }
        return
      }
      // 不在当前图谱中：拉取其邻域并加入
      try {
        const sub = await getEntitySubgraph(name, 2)
        mergeSubgraph(sub.nodes as KgNode[], sub.links as KgLink[])
        const n = sub.nodes.find((x) => x.name === name)
        if (n) {
          setSelectedNode({
            ...n,
            pagerank: pagerankMap.get(name),
            communityId: communityMap.get(name),
          })
          setPathResult(null)
          setActiveTab('details')
        }
      } catch (e) {
        setSearchError((e as Error).message)
      }
    },
    [nodes, mergeSubgraph, pagerankMap],
  )

  // ==========================
  // 搜索
  // ==========================
  const handleSearch = useCallback(async () => {
    const q = searchQuery.trim()
    if (!q) return
    setSearching(true)
    setSearchError(null)
    setActiveTab('search')
    try {
      const res = await searchEntities(q, 15, 0.4)
      setSearchResults(res)
    } catch (e) {
      setSearchError((e as Error).message)
      setSearchResults([])
    } finally {
      setSearching(false)
    }
  }, [searchQuery])

  // ==========================
  // 最短路径
  // ==========================
  const handleFindPath = useCallback(async () => {
    const s = pathStart.trim()
    const t = pathEnd.trim()
    if (!s || !t) return
    setFindingPath(true)
    setPathError(null)
    setActiveTab('path')
    try {
      const res = await getShortestPath(s, t)
      setPathResult(res)
      // 若路径节点不在图中，拉取并入图
      const missing = res.path.filter(
        (name) => !nodes.some((n) => n.name === name),
      )
      if (missing.length > 0) {
        try {
          const sub = await getMultiEntitySubgraph(missing, 1)
          mergeSubgraph(sub.nodes as KgNode[], sub.links as KgLink[])
        } catch {
          // ignore
        }
      }
    } catch (e) {
      setPathError((e as Error).message)
      setPathResult(null)
    } finally {
      setFindingPath(false)
    }
  }, [pathStart, pathEnd, nodes, mergeSubgraph])

  // ==========================
  // 高亮集合计算
  // ==========================
  const neighborLinks = useMemo<KgLink[]>(() => {
    if (!selectedNode) return []
    return links.filter(
      (l) => l.source === selectedNode.name || l.target === selectedNode.name,
    )
  }, [links, selectedNode])

  const activeNodeIds = useMemo<Set<string> | null>(() => {
    if (pathResult && pathResult.path.length > 0) {
      return new Set(pathResult.path)
    }
    if (selectedNode) {
      const s = new Set<string>([selectedNode.id])
      for (const l of neighborLinks) {
        s.add(l.source)
        s.add(l.target)
      }
      return s
    }
    return null
  }, [pathResult, selectedNode, neighborLinks])

  const activeLinkKeys = useMemo<Set<string> | null>(() => {
    if (pathResult && pathResult.path.length > 0) {
      const s = new Set<string>()
      for (let i = 0; i < pathResult.path.length - 1; i++) {
        s.add(linkKeyOf(pathResult.path[i], pathResult.path[i + 1]))
      }
      return s
    }
    if (selectedNode) {
      const s = new Set<string>()
      for (const l of neighborLinks) {
        s.add(linkKeyOf(l.source, l.target))
      }
      return s
    }
    return null
  }, [pathResult, selectedNode, neighborLinks])

  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null)
    setPathResult(null)
  }, [])

  // ==========================
  // 渲染
  // ==========================
  const isLoading = loading || refreshing

  return (
    <div className="flex h-[calc(100dvh-6rem)] flex-col gap-3 md:h-[calc(100dvh-7rem)]">
      {/* 顶部：统计卡片 + 搜索框 */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <StatCard
          label="总实体数"
          value={stats?.total_entities ?? '—'}
        />
        <StatCard
          label="总关系数"
          value={stats?.total_relations ?? '—'}
        />
        <StatCard
          label="总边数"
          value={stats?.total_edges ?? '—'}
        />
        <StatCard
          label="平均度数"
          value={stats?.avg_degree ?? '—'}
        />
        <StatCard
          label="显示节点"
          value={nodes.length}
          hint={`边 ${links.length}`}
        />
        <div className="col-span-2 flex items-center md:col-span-3 lg:col-span-2">
          <div className="flex w-full gap-2">
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSearch()
              }}
              placeholder="搜索实体（如：宫崎骏、盗梦空间）…"
              className="h-10"
            />
            <Button
              onClick={handleSearch}
              disabled={searching || !searchQuery.trim()}
              className="h-10 shrink-0"
            >
              {searching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              搜索
            </Button>
          </div>
        </div>
      </div>

      {/* 主体：图谱画布 + 侧边面板 */}
      <div className="flex min-h-0 flex-1 gap-3">
        {/* 画布 */}
        <div className="relative min-w-0 flex-1 overflow-hidden rounded-lg border bg-card">
          <ForceGraph
            ref={graphRef}
            nodes={nodes}
            links={links}
            theme={theme}
            selectedNodeId={selectedNode?.id ?? null}
            activeNodeIds={activeNodeIds}
            activeLinkKeys={activeLinkKeys}
            onNodeClick={handleNodeClick}
            onBackgroundClick={handleBackgroundClick}
            className="h-full w-full"
          />

          {/* 顶部标题条 */}
          <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Network className="h-3.5 w-3.5" />
            <span>知识图谱可视化</span>
          </div>

          {/* 控制按钮 */}
          <div className="absolute right-3 top-3 flex flex-col gap-1.5">
            <Button
              variant="secondary"
              size="icon"
              className="h-8 w-8"
              onClick={() => graphRef.current?.zoomIn()}
              title="放大"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button
              variant="secondary"
              size="icon"
              className="h-8 w-8"
              onClick={() => graphRef.current?.zoomOut()}
              title="缩小"
              aria-label="缩小图谱"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button
              variant="secondary"
              size="icon"
              className="h-8 w-8"
              onClick={() => graphRef.current?.resetView()}
              title="重置视图"
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
            <Button
              variant="secondary"
              size="icon"
              className="h-8 w-8"
              onClick={() => graphRef.current?.exportPng()}
              title="导出图片"
            >
              <Download className="h-4 w-4" />
            </Button>
            <Button
              variant="secondary"
              size="icon"
              className="h-8 w-8"
              onClick={() => loadInitial()}
              title="刷新图谱"
              disabled={isLoading}
            >
              <RefreshCw
                className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`}
              />
            </Button>
          </div>

          {/* 折叠侧栏按钮 */}
          <Button
            variant="secondary"
            size="icon"
            className="absolute bottom-3 right-3 h-8 w-8"
            onClick={() => setPanelCollapsed((v) => !v)}
            title={panelCollapsed ? '展开侧边面板' : '折叠侧边面板'}
            aria-label={panelCollapsed ? '展开侧边面板' : '折叠侧边面板'}
            aria-expanded={!panelCollapsed}
          >
            {panelCollapsed ? (
              <PanelRightOpen className="h-4 w-4" />
            ) : (
              <PanelRightClose className="h-4 w-4" />
            )}
          </Button>

          {/* 加载/错误遮罩 */}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/60 backdrop-blur-sm">
              <div className="flex flex-col items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-6 w-6 animate-spin" />
                <span>{refreshing ? '刷新图谱中…' : '加载图谱中…'}</span>
              </div>
            </div>
          )}
          {loadError && !isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80 p-6">
              <div className="flex max-w-sm flex-col items-center gap-3 text-center">
                <AlertCircle className="h-8 w-8 text-destructive" />
                <p className="text-sm text-foreground">{loadError}</p>
                <Button size="sm" onClick={() => loadInitial()}>
                  <RefreshCw className="h-4 w-4" />
                  重试
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* 侧边面板：移动端固定底部抽屉，桌面端右侧固定卡片 */}
        {!panelCollapsed && (
          <Card className="flex shrink-0 flex-col overflow-hidden max-md:fixed max-md:inset-x-0 max-md:bottom-0 max-md:top-16 max-md:z-40 max-md:w-full max-md:rounded-b-none max-md:border-x-0 max-md:border-b-0 md:w-80">
            <CardContent className="flex min-h-0 flex-1 flex-col p-3">
              <GraphSidePanel
                selectedNode={selectedNode}
                neighborLinks={neighborLinks}
                pagerankMap={pagerankMap}
                communityMap={communityMap}
                searchQuery={searchQuery}
                onSearchQueryChange={setSearchQuery}
                onSearch={handleSearch}
                searchResults={searchResults}
                searching={searching}
                searchError={searchError}
                onFocusEntity={handleFocusEntity}
                pathStart={pathStart}
                pathEnd={pathEnd}
                onPathStartChange={setPathStart}
                onPathEndChange={setPathEnd}
                onFindPath={handleFindPath}
                pathResult={pathResult}
                pathError={pathError}
                findingPath={findingPath}
                activeTab={activeTab}
                onActiveTabChange={setActiveTab}
              />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
