import {
  ComposedChart,
  Line,
  Area,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import type { ApiForecastVsActual } from '../../types'
import { fmt } from '../../lib/utils'

interface Props {
  data: ApiForecastVsActual[]
}

interface ChartPoint {
  time: string
  predicted: number
  actual: number
  ci_lower: number
  ci_upper: number
}

interface TooltipPayloadEntry {
  dataKey: string
  value: number
  payload: ChartPoint
  color: string
  name: string
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null
  const p = payload[0]?.payload
  if (!p) return null
  const err = Math.abs(p.actual - p.predicted)
  const pct = p.actual > 0 ? ((err / p.actual) * 100).toFixed(1) : '—'
  return (
    <div className="bg-elevated border border-border rounded px-3 py-2 text-xs shadow-lg min-w-[160px]">
      <div className="text-muted mb-1.5">{p.time}</div>
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-sm bg-accent inline-block" />
        <span className="text-muted">Прогноз:</span>
        <span className="font-mono font-semibold text-foreground">{fmt(p.predicted)}</span>
      </div>
      <div className="flex items-center gap-2 mt-0.5">
        <span className="w-2 h-2 rounded-full bg-status-green inline-block" />
        <span className="text-muted">Факт:</span>
        <span className="font-mono font-semibold text-foreground">{fmt(p.actual)}</span>
      </div>
      <div className="flex items-center gap-2 mt-0.5">
        <span className="w-2 h-2 rounded-sm bg-muted inline-block opacity-40" />
        <span className="text-muted">ДИ:</span>
        <span className="font-mono text-muted">{fmt(p.ci_lower)} – {fmt(p.ci_upper)}</span>
      </div>
      <div className="mt-1 pt-1 border-t border-border text-muted">
        Ошибка: {fmt(err)} ({pct}%)
      </div>
    </div>
  )
}

function formatTimeTick(ts: string) {
  const d = new Date(ts)
  const day = String(d.getDate()).padStart(2, '0')
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const hours = String(d.getHours()).padStart(2, '0')
  const mins = String(d.getMinutes()).padStart(2, '0')
  return `${day}.${month} ${hours}:${mins}`
}

export function ForecastVsActualChart({ data }: Props) {
  const chartData: ChartPoint[] = data.map(d => ({
    time: d.timestamp,
    predicted: d.predicted,
    actual: d.actual,
    ci_lower: d.ci_lower,
    ci_upper: d.ci_upper,
  }))

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
        <XAxis
          dataKey="time"
          tick={{ fill: '#7D8590', fontSize: 10 }}
          axisLine={{ stroke: '#30363D' }}
          tickLine={false}
          tickFormatter={formatTimeTick}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: '#7D8590', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={44}
          tickFormatter={v => fmt(v as number)}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          formatter={(value: string) => {
            const labels: Record<string, string> = {
              ci_upper: 'Доверительный интервал',
              predicted: 'Прогноз',
              actual: 'Факт',
            }
            return labels[value] || value
          }}
        />
        {/* CI band — rendered as two stacked areas */}
        <Area
          dataKey="ci_upper"
          stroke="none"
          fill="#FFFFFF"
          fillOpacity={0.14}
          isAnimationActive={false}
        />
        <Area
          dataKey="ci_lower"
          stroke="none"
          fill="#0D1117"
          fillOpacity={1}
          isAnimationActive={false}
          legendType="none"
        />
        {/* Forecast line */}
        <Line
          type="monotone"
          dataKey="predicted"
          stroke="#58A6FF"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        {/* Actual points */}
        <Scatter
          dataKey="actual"
          fill="#3FB950"
          r={3}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
