import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
} from 'recharts'
import type { ForecastPoint } from '../../types'
import { fmt } from '../../lib/utils'

interface ForecastChartProps {
  data: ForecastPoint[]
  /** Scale factor 0-1 to proportionally filter by route readyToShip */
  scale?: number
}

interface ChartPoint {
  time: string
  value: number
  errorY: [number, number]
}

interface TooltipPayloadEntry {
  dataKey: string
  value: number
  payload: ChartPoint
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayloadEntry[]
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null
  const p = payload[0]?.payload
  if (!p) return null
  return (
    <div className="bg-elevated border border-border rounded px-3 py-2 text-xs shadow-lg">
      <div className="text-muted mb-1">{label}</div>
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-sm bg-accent inline-block" />
        <span className="text-muted">Прогноз:</span>
        <span className="font-mono font-semibold text-foreground">{fmt(p.value)}</span>
      </div>
      <div className="flex items-center gap-2 mt-0.5">
        <span className="w-2 h-2 rounded-sm bg-muted inline-block opacity-60" />
        <span className="text-muted">Диапазон:</span>
        <span className="font-mono text-muted">
          {fmt(p.value - p.errorY[0])} – {fmt(p.value + p.errorY[1])}
        </span>
      </div>
    </div>
  )
}

export function ForecastChart({ data, scale = 1 }: ForecastChartProps) {
  const chartData: ChartPoint[] = data.map(d => {
    const v = Math.round(d.value * scale)
    const lo = Math.round(d.lower * scale)
    const hi = Math.round(d.upper * scale)
    return {
      time: d.time,
      value: v,
      errorY: [v - lo, hi - v],
    }
  })

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 8 }} barCategoryGap="28%">
        <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
        <XAxis
          dataKey="time"
          tick={{ fill: '#7D8590', fontSize: 10 }}
          axisLine={{ stroke: '#30363D' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#7D8590', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={36}
          tickFormatter={v => fmt(v as number)}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: '#30363D', fillOpacity: 0.5 }} />
        <Bar dataKey="value" fill="#58A6FF" fillOpacity={0.85} radius={[3, 3, 0, 0]}>
          <ErrorBar dataKey="errorY" width={4} strokeWidth={1.5} stroke="#7D8590" direction="y" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
