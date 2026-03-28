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
  const graph = useMemo<ProcessedGraph>(() => {
    const gen = d3Sankey<SankeyNodeDatum, SankeyLinkDatum>()
      .nodeId(d => d.id)
      .nodeWidth(10)
      .nodePadding(16)
      .extent([
        [1, 1],
        [width - 1, height - 1],
      ])

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return gen({
      nodes: data.nodes.map(n => ({ ...n })),
      links: data.links.map(l => ({ ...l })),
    }) as unknown as ProcessedGraph
  }, [data, width, height])

  const pathGen = sankeyLinkHorizontal()

  const isBottleneck = (link: LayoutLink, idx: number): boolean => {
    if (idx === 0) return false
    const prev = graph.links[idx - 1]
    return prev.value > 0 && link.value / prev.value < 0.7
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
        const cy = node.y0 + nodeHeight / 2
        const isLeft = node.x0 < width / 2

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
              x={isLeft ? node.x1 + 7 : node.x0 - 7}
              y={cy - 5}
              textAnchor={isLeft ? 'start' : 'end'}
              dominantBaseline="middle"
              fill="#E6EDF3"
              fontSize={10}
              fontFamily="Inter, sans-serif"
              fontWeight={500}
            >
              {node.label}
            </text>
            {/* Value */}
            <text
              x={isLeft ? node.x1 + 7 : node.x0 - 7}
              y={cy + 7}
              textAnchor={isLeft ? 'start' : 'end'}
              dominantBaseline="middle"
              fill="#7D8590"
              fontSize={9}
              fontFamily="JetBrains Mono, monospace"
            >
              {fmt(node.value)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
