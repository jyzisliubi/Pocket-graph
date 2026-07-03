import type { ElementType, ReactNode } from 'react'
import {
  ArrowRight,
  Hash,
  Layers,
  Route,
  Search,
  Sparkles,
  Users,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { communityColor, type KgLink, type KgNode } from './ForceGraph'
import type { EntitySearchResult, ShortestPathResponse } from '@/types/api'

export interface GraphSidePanelProps {
  /** 当前选中节点 */
  selectedNode: KgNode | null
  /** 与选中节点相连的边（用于展示关联关系） */
  neighborLinks: KgLink[]
  /** PageRank 分数表 */
  pagerankMap: Map<string, number>
  /** 社区 ID 表 */
  communityMap: Map<string, number>

  // 搜索
  searchQuery: string
  onSearchQueryChange: (q: string) => void
  onSearch: () => void
  searchResults: EntitySearchResult[]
  searching: boolean
  searchError: string | null
  onFocusEntity: (name: string) => void

  // 最短路径
  pathStart: string
  pathEnd: string
  onPathStartChange: (v: string) => void
  onPathEndChange: (v: string) => void
  onFindPath: () => void
  pathResult: ShortestPathResponse | null
  pathError: string | null
  findingPath: boolean

  /** 当前激活的 Tab */
  activeTab: string
  onActiveTabChange: (tab: string) => void
}

/** 信息行：图标 + 标签 + 值 */
function InfoRow({
  icon: Icon,
  label,
  children,
}: {
  icon: ElementType
  label: string
  children: ReactNode
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </span>
      <span className="text-sm font-medium">{children}</span>
    </div>
  )
}

export function GraphSidePanel(props: GraphSidePanelProps) {
  const {
    selectedNode,
    neighborLinks,
    pagerankMap,
    communityMap,
    searchQuery,
    onSearchQueryChange,
    onSearch,
    searchResults,
    searching,
    searchError,
    onFocusEntity,
    pathStart,
    pathEnd,
    onPathStartChange,
    onPathEndChange,
    onFindPath,
    pathResult,
    pathError,
    findingPath,
    activeTab,
    onActiveTabChange,
  } = props

  const pr = selectedNode
    ? pagerankMap.get(selectedNode.name) ?? selectedNode.pagerank
    : undefined
  const cid = selectedNode
    ? communityMap.get(selectedNode.name) ?? selectedNode.communityId
    : undefined

  return (
    <Tabs value={activeTab} onValueChange={onActiveTabChange} className="flex h-full flex-col">
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="details" className="text-xs">
          实体详情
        </TabsTrigger>
        <TabsTrigger value="search" className="text-xs">
          搜索
        </TabsTrigger>
        <TabsTrigger value="path" className="text-xs">
          路径
        </TabsTrigger>
      </TabsList>

      {/* ========================== 实体详情 ========================== */}
      <TabsContent
        value="details"
        className="mt-3 flex-1 overflow-y-auto pr-1"
      >
        {!selectedNode ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
            <Sparkles className="h-8 w-8 opacity-40" />
            <span>点击图谱中的节点查看实体详情</span>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="rounded-md border bg-muted/40 p-3">
              <div className="mb-1 flex items-center gap-2">
                {cid !== undefined && (
                  <span
                    className="inline-block h-3 w-3 rounded-full"
                    style={{ backgroundColor: communityColor(cid) }}
                  />
                )}
                <span className="text-base font-semibold leading-tight">
                  {selectedNode.name}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">实体 ID：{selectedNode.id}</p>
            </div>

            <div className="divide-y">
              <InfoRow icon={Hash} label="度数（Degree）">
                <Badge variant="secondary">{selectedNode.degree}</Badge>
              </InfoRow>
              <InfoRow icon={Sparkles} label="PageRank 分数">
                {pr !== undefined ? pr.toFixed(5) : '—'}
              </InfoRow>
              <InfoRow icon={Layers} label="社区 ID">
                {cid !== undefined ? (
                  <Badge
                    variant="outline"
                    style={{
                      color: communityColor(cid),
                      borderColor: communityColor(cid),
                    }}
                  >
                    {cid}
                  </Badge>
                ) : (
                  '—'
                )}
              </InfoRow>
            </div>

            <div>
              <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Users className="h-3.5 w-3.5" />
                关联关系（{neighborLinks.length}）
              </div>
              {neighborLinks.length === 0 ? (
                <p className="text-xs text-muted-foreground">暂无关联边</p>
              ) : (
                <ul className="space-y-1.5">
                  {neighborLinks.slice(0, 50).map((l, i) => {
                    const other =
                      l.source === selectedNode.name ? l.target : l.source
                    return (
                      <li
                        key={`${l.source}-${l.target}-${l.relation}-${i}`}
                        className="flex flex-wrap items-center gap-1 rounded-md border bg-background px-2 py-1.5 text-xs"
                      >
                        <button
                          className="font-medium text-primary hover:underline"
                          onClick={() => onFocusEntity(other)}
                        >
                          {other}
                        </button>
                        <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" />
                        <span className="truncate text-muted-foreground">
                          {l.relation}
                        </span>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          </div>
        )}
      </TabsContent>

      {/* ========================== 实体搜索 ========================== */}
      <TabsContent
        value="search"
        className="mt-3 flex-1 overflow-y-auto pr-1"
      >
        <div className="space-y-3">
          <div className="flex gap-2">
            <Input
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onSearch()
              }}
              placeholder="输入关键词搜索实体…"
              className="h-9"
            />
            <Button
              size="sm"
              onClick={onSearch}
              disabled={searching || !searchQuery.trim()}
              className="h-9 shrink-0"
            >
              <Search className="h-4 w-4" />
              搜索
            </Button>
          </div>

          {searchError && (
            <p className="text-xs text-destructive">{searchError}</p>
          )}

          {searching ? (
            <p className="text-xs text-muted-foreground">搜索中…</p>
          ) : searchResults.length > 0 ? (
            <ul className="space-y-1.5">
              {searchResults.map((r) => (
                <li key={r.entity}>
                  <button
                    className="flex w-full items-center justify-between rounded-md border bg-background px-2.5 py-2 text-left text-sm transition-colors hover:bg-accent"
                    onClick={() => onFocusEntity(r.entity)}
                  >
                    <span className="truncate font-medium">{r.entity}</span>
                    <Badge variant="secondary" className="ml-2 shrink-0">
                      度 {r.degree}
                    </Badge>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            !searchError &&
            searchQuery.trim() && (
              <p className="text-xs text-muted-foreground">无匹配结果</p>
            )
          )}
          {!searchQuery.trim() && (
            <p className="text-xs text-muted-foreground">
              输入实体名称关键词，点击结果可在图谱中聚焦对应节点。
            </p>
          )}
        </div>
      </TabsContent>

      {/* ========================== 最短路径 ========================== */}
      <TabsContent
        value="path"
        className="mt-3 flex-1 overflow-y-auto pr-1"
      >
        <div className="space-y-3">
          <div className="space-y-2">
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">
                起点实体
              </label>
              <Input
                value={pathStart}
                onChange={(e) => onPathStartChange(e.target.value)}
                placeholder="起点实体名称"
                className="h-9"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">
                终点实体
              </label>
              <Input
                value={pathEnd}
                onChange={(e) => onPathEndChange(e.target.value)}
                placeholder="终点实体名称"
                className="h-9"
              />
            </div>
            <Button
              size="sm"
              className="w-full"
              onClick={onFindPath}
              disabled={findingPath || !pathStart.trim() || !pathEnd.trim()}
            >
              <Route className="h-4 w-4" />
              查询最短路径
            </Button>
          </div>

          {pathError && (
            <p className="text-xs text-destructive">{pathError}</p>
          )}

          {pathResult && (
            <div className="rounded-md border bg-muted/40 p-3">
              <p className="mb-2 text-xs text-muted-foreground">
                路径长度：
                <span className="font-semibold text-foreground">
                  {pathResult.length}
                </span>
              </p>
              {pathResult.path.length > 0 ? (
                <ol className="space-y-1">
                  {pathResult.path.map((node, i) => (
                    <li key={`${node}-${i}`} className="flex items-center gap-1.5 text-xs">
                      <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary" />
                      <button
                        className="font-medium text-primary hover:underline"
                        onClick={() => onFocusEntity(node)}
                      >
                        {node}
                      </button>
                      {i < pathResult.path.length - 1 && (
                        <ArrowRight className="ml-auto h-3 w-3 text-muted-foreground" />
                      )}
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-xs text-muted-foreground">
                  两实体间不存在可达路径
                </p>
              )}
            </div>
          )}

          {!pathResult && !pathError && (
            <p className="text-xs text-muted-foreground">
              选择起点和终点实体，查询它们在图谱中的最短路径并高亮显示。
            </p>
          )}
        </div>
      </TabsContent>
    </Tabs>
  )
}
