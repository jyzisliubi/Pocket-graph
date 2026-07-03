import * as d3 from 'd3'
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'
import type { GraphLink, GraphNode } from '@/types/api'
import type { Theme } from '@/lib/theme'

/**
 * 知识图谱节点：在 GraphNode 基础上扩展 PageRank 分数与社区 ID，
 * 用于节点大小/颜色映射。
 */
export interface KgNode extends GraphNode {
  pagerank?: number
  communityId?: number
}

export type KgLink = GraphLink

/** 仿真节点（携带 d3 仿真坐标） */
type SimNode = KgNode & d3.SimulationNodeDatum
/** 仿真边（d3 会把 source/target 解析为节点对象引用） */
type SimLink = d3.SimulationLinkDatum<SimNode> & {
  relation: string
  source: string | SimNode
  target: string | SimNode
}

/** 暴露给父组件的命令式方法 */
export interface ForceGraphHandle {
  /** 重置视图（缩放/平移回到初始） */
  resetView: () => void
  /** 放大 */
  zoomIn: () => void
  /** 缩小 */
  zoomOut: () => void
  /** 导出当前图谱为 PNG 图片 */
  exportPng: (filename?: string) => void
}

export interface ForceGraphProps {
  nodes: KgNode[]
  links: KgLink[]
  theme: Theme
  /** 当前选中节点 ID */
  selectedNodeId: string | null
  /** 需高亮的节点 ID 集合（选中节点及其邻居，或路径节点） */
  activeNodeIds: Set<string> | null
  /** 需高亮的边 key 集合（key = 排序后的 "a|b"） */
  activeLinkKeys: Set<string> | null
  /** 点击节点回调 */
  onNodeClick: (node: KgNode) => void
  /** 点击空白背景回调（用于取消选中） */
  onBackgroundClick?: () => void
  className?: string
}

// ==========================
// 颜色与尺寸常量
// ==========================

/** 社区配色（在深浅背景下均可见的明亮色板） */
const COMMUNITY_COLORS = [
  '#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa',
  '#fb7185', '#22d3ee', '#4ade80', '#facc15', '#c084fc',
  '#f87171', '#38bdf8',
]

/** 主题相关颜色 */
const THEME_COLORS: Record<
  Theme,
  {
    bg: string
    link: string
    linkActive: string
    nodeStroke: string
    text: string
    textDim: string
  }
> = {
  dark: {
    bg: '#0a0a0b',
    link: '#3f4756',
    linkActive: '#e2e8f0',
    nodeStroke: '#0a0a0b',
    text: '#e2e8f0',
    textDim: '#64748b',
  },
  light: {
    bg: '#ffffff',
    link: '#cbd5e1',
    linkActive: '#334155',
    nodeStroke: '#ffffff',
    text: '#1e293b',
    textDim: '#94a3b8',
  },
}

/** 根据社区 ID 取色（导出供侧边栏等复用） */
export function communityColor(id: number | undefined): string {
  if (id === undefined || id === null) return '#94a3b8'
  const n = COMMUNITY_COLORS.length
  return COMMUNITY_COLORS[((id % n) + n) % n]
}

/** 由无向两端生成边 key */
function linkKeyOf(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`
}

/** 取节点 ID（兼容 d3 将 source/target 解析为对象的情况） */
function nodeIdOf(d: string | SimNode): string {
  return typeof d === 'string' ? d : d.id
}

/** 计算节点半径：按 degree 平方根缩放，并叠加 PageRank 加成 */
function nodeRadius(
  d: KgNode,
  maxDegree: number,
  maxPr: number,
): number {
  const deg = d.degree ?? 0
  const degR = 5 + (maxDegree > 0 ? Math.sqrt(deg / maxDegree) : 0) * 14
  const prR =
    d.pagerank && maxPr > 0 ? (d.pagerank / maxPr) * 6 : 0
  return degR + prR
}

/** 内部结构状态：保存 d3 选择集与仿真，供样式 effect 复用 */
interface GraphState {
  simulation: d3.Simulation<SimNode, SimLink>
  linkSel: d3.Selection<SVGLineElement, SimLink, SVGGElement, unknown>
  nodeSel: d3.Selection<SVGCircleElement, SimNode, SVGGElement, unknown>
  labelSel: d3.Selection<SVGTextElement, SimNode, SVGGElement, unknown>
  linkLabelSel: d3.Selection<SVGTextElement, SimLink, SVGGElement, unknown>
  bgRect: d3.Selection<SVGRectElement, unknown, null, undefined>
  glowFilter: d3.Selection<SVGFilterElement, unknown, null, undefined>
  simNodes: SimNode[]
  simLinks: SimLink[]
  maxDegree: number
  maxPr: number
}

export const ForceGraph = forwardRef<ForceGraphHandle, ForceGraphProps>(
  function ForceGraph(props, ref) {
    const {
      nodes,
      links,
      theme,
      selectedNodeId,
      activeNodeIds,
      activeLinkKeys,
      onNodeClick,
      onBackgroundClick,
      className,
    } = props

    const containerRef = useRef<HTMLDivElement>(null)
    const svgRef = useRef<SVGSVGElement>(null)
    const stateRef = useRef<GraphState | null>(null)
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
    const svgSelRef = useRef<d3.Selection<SVGSVGElement, unknown, null, undefined> | null>(null)

    // 回调用 ref 保存最新值，避免重建仿真
    const onNodeClickRef = useRef(onNodeClick)
    const onBackgroundClickRef = useRef(onBackgroundClick)
    onNodeClickRef.current = onNodeClick
    onBackgroundClickRef.current = onBackgroundClick

    // 节点历史坐标缓存：跨重建保留布局，主题/尺寸切换时不抖动
    const positionsRef = useRef<Map<string, { x?: number; y?: number; vx?: number; vy?: number }>>(
      new Map(),
    )

    // 测量容器尺寸
    const [size, setSize] = useState({ width: 800, height: 600 })
    useEffect(() => {
      const el = containerRef.current
      if (!el) return
      const update = () => {
        const r = el.getBoundingClientRect()
        setSize({
          width: Math.max(320, r.width),
          height: Math.max(320, r.height),
        })
      }
      update()
      const ro = new ResizeObserver(update)
      ro.observe(el)
      return () => ro.disconnect()
    }, [])

    const { width, height } = size

    // ==========================
    // 结构 effect：构建 SVG 与仿真（仅在数据/尺寸变化时重建）
    // ==========================
    useEffect(() => {
      if (!svgRef.current) return
      const svgEl = svgRef.current
      const svg = d3.select(svgEl)
      svgSelRef.current = svg

      // 清空旧内容
      svg.selectAll('*').remove()
      svg.attr('width', width).attr('height', height)

      const colors = THEME_COLORS[theme]

      // defs：发光滤镜（深色模式）
      const defs = svg.append('defs')
      const glowFilter = defs
        .append('filter')
        .attr('id', 'kg-glow')
        .attr('x', '-50%')
        .attr('y', '-50%')
        .attr('width', '200%')
        .attr('height', '200%')
      glowFilter
        .append('feGaussianBlur')
        .attr('stdDeviation', 3.5)
        .attr('result', 'blur')
      const merge = glowFilter.append('feMerge')
      merge.append('feMergeNode').attr('in', 'blur')
      merge.append('feMergeNode').attr('in', 'SourceGraphic')

      // 背景矩形（用于点击空白取消选中 + 导出底色）
      const bgRect = svg
        .append('rect')
        .attr('class', 'kg-bg')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', colors.bg)
        .style('cursor', 'grab')

      // 主图层（受 zoom transform 控制）
      const g = svg.append('g').attr('class', 'kg-stage')

      // 边层 / 边标签层 / 节点层 / 节点标签层
      const linkLayer = g.append('g').attr('class', 'kg-links')
      const linkLabelLayer = g.append('g').attr('class', 'kg-link-labels')
      const nodeLayer = g.append('g').attr('class', 'kg-nodes')
      const labelLayer = g.append('g').attr('class', 'kg-labels')

      // 准备仿真数据（拷贝，避免污染 props；并从缓存恢复坐标）
      const maxDegree = d3.max(nodes, (d) => d.degree) ?? 1
      const maxPr = d3.max(nodes, (d) => d.pagerank ?? 0) ?? 0
      const simNodes: SimNode[] = nodes.map((n) => {
        const prev = positionsRef.current.get(n.id)
        return { ...n, x: prev?.x, y: prev?.y, vx: prev?.vx, vy: prev?.vy }
      })
      const simLinks: SimLink[] = links.map((l) => ({
        ...l,
        source: l.source,
        target: l.target,
      }))

      // 节点半径辅助
      const rOf = (d: SimNode) => nodeRadius(d, maxDegree, maxPr)

      // 边
      const linkSel = linkLayer
        .selectAll<SVGLineElement, SimLink>('line')
        .data(simLinks)
        .join('line')
        .attr('stroke', colors.link)
        .attr('stroke-width', 1.2)
        .attr('stroke-opacity', 0.5)

      // 边关系标签（默认隐藏）
      const linkLabelSel = linkLabelLayer
        .selectAll<SVGTextElement, SimLink>('text')
        .data(simLinks)
        .join('text')
        .attr('text-anchor', 'middle')
        .attr('dy', -3)
        .attr('font-size', 9)
        .attr('fill', colors.textDim)
        .attr('pointer-events', 'none')
        .attr('opacity', 0)
        .text((d) => d.relation)

      // 节点圆
      const nodeSel = nodeLayer
        .selectAll<SVGCircleElement, SimNode>('circle')
        .data(simNodes, (d) => d.id)
        .join('circle')
        .attr('r', rOf)
        .attr('fill', (d) => communityColor(d.communityId))
        .attr('stroke', colors.nodeStroke)
        .attr('stroke-width', 1.5)
        .style('cursor', 'pointer')

      // 节点标签（默认按度数显隐，由样式 effect 控制）
      const labelSel = labelLayer
        .selectAll<SVGTextElement, SimNode>('text')
        .data(simNodes, (d) => d.id)
        .join('text')
        .attr('text-anchor', 'middle')
        .attr('dy', (d) => -rOf(d) - 4)
        .attr('font-size', 11)
        .attr('fill', colors.text)
        .attr('pointer-events', 'none')
        .text((d) => d.name)
        .attr('opacity', (d) => (d.degree >= 6 ? 0.9 : 0))

      // ==========================
      // 仿真
      // ==========================
      const simulation = d3
        .forceSimulation<SimNode>(simNodes)
        .force(
          'link',
          d3
            .forceLink<SimNode, SimLink>(simLinks)
            .id((d) => d.id)
            .distance((l) => {
              const s = typeof l.source === 'string' ? null : l.source
              const t = typeof l.target === 'string' ? null : l.target
              const dr = s && t ? (s.degree + t.degree) / 2 : 5
              return 30 + 120 / Math.max(1, Math.sqrt(dr))
            })
            .strength(0.35),
        )
        .force('charge', d3.forceManyBody().strength(-220))
        .force(
          'collision',
          d3.forceCollide<SimNode>().radius((d) => rOf(d) + 4),
        )
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force(
          'x',
          d3.forceX(width / 2).strength(0.04),
        )
        .force(
          'y',
          d3.forceY(height / 2).strength(0.04),
        )
        .alpha(1)
        .alphaDecay(0.025)

      simulation.on('tick', () => {
        // 缓存坐标
        for (const n of simNodes) {
          positionsRef.current.set(n.id, {
            x: n.x,
            y: n.y,
            vx: n.vx,
            vy: n.vy,
          })
        }
        linkSel
          .attr('x1', (d) => (d.source as SimNode).x ?? 0)
          .attr('y1', (d) => (d.source as SimNode).y ?? 0)
          .attr('x2', (d) => (d.target as SimNode).x ?? 0)
          .attr('y2', (d) => (d.target as SimNode).y ?? 0)
        nodeSel.attr('cx', (d) => d.x ?? 0).attr('cy', (d) => d.y ?? 0)
        labelSel.attr('x', (d) => d.x ?? 0).attr('y', (d) => d.y ?? 0)
        linkLabelSel
          .attr('x', (d) => {
            const sx = (d.source as SimNode).x ?? 0
            const tx = (d.target as SimNode).x ?? 0
            return (sx + tx) / 2
          })
          .attr('y', (d) => {
            const sy = (d.source as SimNode).y ?? 0
            const ty = (d.target as SimNode).y ?? 0
            return (sy + ty) / 2
          })
      })

      // ==========================
      // 缩放与平移
      // ==========================
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.15, 6])
        .on('zoom', (event) => {
          g.attr('transform', event.transform.toString())
        })
      zoomRef.current = zoom
      svg.call(zoom)
      // 初始以一个略微缩小的视角居中
      svg.call(zoom.transform, d3.zoomIdentity)

      // 背景平移光标反馈 + 拖拽阈值判断（避免平移后误触发取消选中）
      const pointerDown = { x: 0, y: 0, moved: false }
      bgRect
        .on('pointerdown', (event: MouseEvent) => {
          pointerDown.x = event.clientX
          pointerDown.y = event.clientY
          pointerDown.moved = false
          svg.style('cursor', 'grabbing')
        })
        .on('pointermove', (event: MouseEvent) => {
          if (event.buttons === 0) return
          const dx = event.clientX - pointerDown.x
          const dy = event.clientY - pointerDown.y
          if (dx * dx + dy * dy > 25) pointerDown.moved = true
        })
        .on('pointerup', () => svg.style('cursor', 'grab'))
        .on('click', () => {
          // 平移超过阈值则视为拖拽，不触发取消选中
          if (pointerDown.moved) return
          onBackgroundClickRef.current?.()
        })

      // ==========================
      // 节点拖拽（重新布局）
      // ==========================
      const drag = d3
        .drag<SVGCircleElement, SimNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      nodeSel.call(drag)

      // 节点悬停：放大 + 显示标签
      nodeSel
        .on('mouseenter', function (_, d) {
          d3.select(this).attr('stroke-width', 2.5).attr('r', rOf(d) + 2)
          labelSel
            .filter((n) => n.id === d.id)
            .attr('opacity', 1)
        })
        .on('mouseleave', function (_, d) {
          d3.select(this).attr('stroke-width', 1.5).attr('r', rOf(d))
          // 标签显隐交还样式 effect 控制（用 ref 取最新闭包，避免主题切换后用到旧色）
          applyStylesRef.current()
        })
        .on('click', function (event, d) {
          event.stopPropagation()
          onNodeClickRef.current(d)
        })

      // 保存状态供样式 effect 使用
      stateRef.current = {
        simulation,
        linkSel,
        nodeSel,
        labelSel,
        linkLabelSel,
        bgRect,
        glowFilter,
        simNodes,
        simLinks,
        maxDegree,
        maxPr,
      }

      // 应用一次当前高亮与样式
      applyStylesRef.current()

      // 清理：停止仿真
      return () => {
        simulation.stop()
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [nodes, links, width, height])

    // ==========================
    // 样式 effect：主题与高亮（不重建仿真）
    // ==========================
    function applyStyles() {
      const st = stateRef.current
      if (!st) return
      const colors = THEME_COLORS[theme]
      const rOf = (d: SimNode) => nodeRadius(d, st.maxDegree, st.maxPr)

      const hasActive =
        (activeNodeIds && activeNodeIds.size > 0) ||
        (activeLinkKeys && activeLinkKeys.size > 0)
      const isActive = (id: string) => !!activeNodeIds?.has(id)

      // 背景
      st.bgRect.attr('fill', colors.bg)

      // 发光滤镜：仅深色模式启用
      if (theme === 'dark') {
        st.nodeSel.style('filter', 'url(#kg-glow)')
      } else {
        st.nodeSel.style('filter', null)
      }

      // 节点颜色与描边
      st.nodeSel
        .attr('fill', (d) => communityColor(d.communityId))
        .attr('stroke', (d) =>
          d.id === selectedNodeId ? colors.linkActive : colors.nodeStroke,
        )
        .attr('stroke-width', (d) => (d.id === selectedNodeId ? 3 : 1.5))
        .attr('r', (d) => (d.id === selectedNodeId ? rOf(d) + 2 : rOf(d)))
        .style('opacity', (d) => {
          if (!hasActive) return 1
          return isActive(d.id) ? 1 : 0.2
        })

      // 节点标签
      st.labelSel
        .attr('fill', colors.text)
        .attr('dy', (d) => -rOf(d) - 4)
        .attr('opacity', (d) => {
          if (d.id === selectedNodeId || isActive(d.id)) return 1
          if (!hasActive && d.degree >= 6) return 0.9
          return 0
        })

      // 边
      st.linkSel
        .attr('stroke', (d) => {
          const k = linkKeyOf(nodeIdOf(d.source), nodeIdOf(d.target))
          return activeLinkKeys?.has(k) ? colors.linkActive : colors.link
        })
        .attr('stroke-width', (d) => {
          const k = linkKeyOf(nodeIdOf(d.source), nodeIdOf(d.target))
          return activeLinkKeys?.has(k) ? 2.4 : 1.2
        })
        .attr('stroke-opacity', (d) => {
          if (!hasActive) return 0.5
          const k = linkKeyOf(nodeIdOf(d.source), nodeIdOf(d.target))
          return activeLinkKeys?.has(k) ? 0.95 : 0.06
        })

      // 边关系标签：仅高亮边显示
      st.linkLabelSel
        .attr('fill', colors.textDim)
        .attr('opacity', (d) => {
          if (!activeLinkKeys) return 0
          const k = linkKeyOf(nodeIdOf(d.source), nodeIdOf(d.target))
          return activeLinkKeys.has(k) ? 1 : 0
        })
    }

    // 始终指向最新 applyStyles 闭包，供结构 effect 中的事件回调调用
    const applyStylesRef = useRef<() => void>(() => {})
    applyStylesRef.current = applyStyles

    useEffect(() => {
      applyStyles()
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [theme, selectedNodeId, activeNodeIds, activeLinkKeys])

    // ==========================
    // 命令式方法
    // ==========================
    useImperativeHandle(ref, () => ({
      resetView() {
        const svg = svgSelRef.current
        const zoom = zoomRef.current
        if (!svg || !zoom) return
        svg
          .transition()
          .duration(500)
          .call(zoom.transform, d3.zoomIdentity)
      },
      zoomIn() {
        const svg = svgSelRef.current
        const zoom = zoomRef.current
        if (!svg || !zoom) return
        zoom.scaleBy(svg.transition().duration(250), 1.35)
      },
      zoomOut() {
        const svg = svgSelRef.current
        const zoom = zoomRef.current
        if (!svg || !zoom) return
        zoom.scaleBy(svg.transition().duration(250), 1 / 1.35)
      },
      exportPng(filename = 'knowledge-graph.png') {
        const svgEl = svgRef.current
        if (!svgEl) return
        const w = width
        const h = height
        const clone = svgEl.cloneNode(true) as SVGSVGElement
        clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        clone.setAttribute('width', String(w))
        clone.setAttribute('height', String(h))
        const xml = new XMLSerializer().serializeToString(clone)
        const svgBlob = new Blob([xml], {
          type: 'image/svg+xml;charset=utf-8',
        })
        const url = URL.createObjectURL(svgBlob)
        const img = new Image()
        img.onload = () => {
          const scale = 2
          const canvas = document.createElement('canvas')
          canvas.width = w * scale
          canvas.height = h * scale
          const ctx = canvas.getContext('2d')
          if (!ctx) {
            URL.revokeObjectURL(url)
            return
          }
          ctx.fillStyle = THEME_COLORS[theme].bg
          ctx.fillRect(0, 0, canvas.width, canvas.height)
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
          URL.revokeObjectURL(url)
          canvas.toBlob((blob) => {
            if (!blob) return
            const a = document.createElement('a')
            a.href = URL.createObjectURL(blob)
            a.download = filename
            a.click()
            URL.revokeObjectURL(a.href)
          }, 'image/png')
        }
        img.onerror = () => URL.revokeObjectURL(url)
        img.src = url
      },
    }))

    return (
      <div ref={containerRef} className={className} style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          style={{ display: 'block', width: '100%', height: '100%' }}
        />
      </div>
    )
  },
)
