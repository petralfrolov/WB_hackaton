import { useMemo } from 'react'
import { sankey as d3Sankey, sankeyLinkHorizontal } from 'd3-sankey'
import type { SankeyData, SankeyNodeDatum, SankeyLinkDatum } from '../../types'
import { fmt } from '../../lib/utils'

interface SankeyChartProps {
  data: SankeyData
  width?: number
  height?: number
}

interface LayoutNode extends SankeyNodeDatum {
  x0: number
  x1: number
  y0: number
  y1: number
  value: number
  index: number
}

interface LayoutLink {
  source: LayoutNode
  target: LayoutNode
  value: number
  width: number
  y0: number
  y1: number
  index: number
}

interface ProcessedGraph {
  nodes: LayoutNode[]
  links: LayoutLink[]
}

export function SankeyChart({ data, width = 620, height = 260 }: SankeyChartProps) {
  const isEmpty = data.nodes.length === 0 || data.links.length === 0

  const graph = useMemo<ProcessedGraph>(() => {
    if (isEmpty) return { nodes: [], links: [] } as unknown as ProcessedGraph

    const gen = d3Sankey<SankeyNodeDatum, SankeyLinkDatum>()
      .nodeId(d => d.id)
      .nodeWidth(12)
      .nodePadding(18)
      .extent([
        [8, 44],
        [width - 16, height - 10],
      ])

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return gen({
      nodes: data.nodes.map(n => ({ ...n })),
      links: data.links.map(l => ({ ...l })),
    }) as unknown as ProcessedGraph
  }, [data, width, height, isEmpty])

  const pathGen = sankeyLinkHorizontal()

  const isBottleneck = (link: LayoutLink, idx: number): boolean => {
    if (idx === 0) return false
    const prev = graph.links[idx - 1]
    return prev.value > 0 && link.value / prev.value < 0.7
  }

  if (isEmpty) {
    return (
      <svg width={width} height={height}>
        <text x={width / 2} y={height / 2} textAnchor="middle" fill="#888" fontSize={14}>
          Нет данных
        </text>
      </svg>
    )
  }

  return (
    <svg
      width={width}
      height={height}
      style={{ overflow: 'visible', display: 'block' }}
    >
      {/* Links */}
      {graph.links.map((link, i) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const d = pathGen(link as unknown as Parameters<typeof pathGen>[0])
        if (!d) return null
        const bottleneck = isBottleneck(link, i)
        return (
          <path
            key={i}
            d={d}
            fill="none"
            stroke={bottleneck ? '#D29922' : '#58A6FF'}
            strokeOpacity={bottleneck ? 0.5 : 0.28}
            strokeWidth={Math.max(1, link.width)}
          />
        )
      })}

      {/* Nodes */}
      {graph.nodes.map(node => {
        const nodeWidth = node.x1 - node.x0
        const nodeHeight = node.y1 - node.y0
        const cx = node.x0 + nodeWidth / 2
        const labelParts = node.label.includes(' ')
          ? [node.label.split(' ').slice(0, -1).join(' '), node.label.split(' ').slice(-1).join('')]
          : [node.label]

        return (
          <g key={node.id}>
            <rect
              x={node.x0}
              y={node.y0}
              width={nodeWidth}
              height={nodeHeight}
              fill="#58A6FF"
              fillOpacity={0.85}
              rx={2}
            />
            {/* Label */}
            <text
              x={cx}
              y={labelParts.length > 1 ? 16 : 22}
              textAnchor="middle"
              fill="#E6EDF3"
              fontSize={10}
              fontFamily="Inter, sans-serif"
              fontWeight={600}
            >
              {labelParts.map((part, idx) => (
                <tspan key={idx} x={cx} dy={idx === 0 ? 0 : 11}>{part}</tspan>
              ))}
            </text>
            {/* Value */}
            <text
              x={cx}
              y={labelParts.length > 1 ? 40 : 34}
              textAnchor="middle"
              fill="#7D8590"
              fontSize={10}
              fontFamily="JetBrains Mono, monospace"
            >
              {fmt((data.nodes.find(n => n.id === node.id)?.value) ?? node.value)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
