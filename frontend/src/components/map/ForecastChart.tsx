import {
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { ForecastPoint } from '../../types'
import { fmt } from '../../lib/utils'

interface ForecastChartProps {
  data: ForecastPoint[]
}

interface ChartPoint {
  time: string
  value: number
  lower: number
  band: number
  upper: number
}

interface TooltipPayloadEntry {
  dataKey: string
  value: number
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayloadEntry[]
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null
  const upper = (payload.find(p => p.dataKey === 'band')?.value ?? 0) + (payload.find(p => p.dataKey === 'lower')?.value ?? 0)
  const lower = payload.find(p => p.dataKey === 'lower')?.value ?? 0
  const value = payload.find(p => p.dataKey === 'value')?.value ?? 0
  return (
    <div className="bg-elevated border border-border rounded px-3 py-2 text-xs shadow-lg">
      <div className="text-muted mb-1">{label}</div>
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-foreground inline-block" />
        <span className="text-muted">Прогноз:</span>
        <span className="font-mono font-semibold text-foreground">{fmt(value)}</span>
      </div>
      <div className="flex items-center gap-2 mt-0.5">
        <span className="w-2 h-2 rounded-full bg-muted inline-block" />
        <span className="text-muted">Диапазон:</span>
        <span className="font-mono text-muted">
          {fmt(lower)} – {fmt(upper)}
        </span>
      </div>
    </div>
  )
}

export function ForecastChart({ data }: ForecastChartProps) {
  const chartData: ChartPoint[] = data.map(d => ({
    time: d.time,
    value: d.value,
    lower: d.lower,
    band: d.upper - d.lower,
    upper: d.upper,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
        <defs>
          <linearGradient id="gradCyan" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#58A6FF" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#58A6FF" stopOpacity={0} />
          </linearGradient>
        </defs>
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
        <Tooltip content={<CustomTooltip />} />
        {/* Confidence band: stacked transparent base + gray fill */}
        <Area
          type="monotone"
          dataKey="lower"
          stackId="ci"
          stroke="none"
          fill="transparent"
          legendType="none"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="band"
          stackId="ci"
          stroke="none"
          fill="#7D8590"
          fillOpacity={0.18}
          legendType="none"
          isAnimationActive={false}
        />
        {/* Forecast area + line */}
        <Area
          type="monotone"
          dataKey="value"
          stroke="#58A6FF"
          strokeWidth={2}
          fill="url(#gradCyan)"
          dot={false}
          activeDot={{ r: 4, fill: '#58A6FF', strokeWidth: 0 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
