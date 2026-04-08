import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Brush,
} from 'recharts'
import type { ApiTimeSeriesMetric } from '../../types'
import { fmt } from '../../lib/utils'

interface Props {
  data: ApiTimeSeriesMetric[]
}

interface TooltipPayloadEntry {
  dataKey: string
  value: number
  color: string
  name: string
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: TooltipPayloadEntry[]; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-elevated border border-border rounded px-3 py-2 text-xs shadow-lg">
      <div className="text-muted mb-1">{formatTimeTick(label || '')}</div>
      {payload.map(p => (
        <div key={p.dataKey} className="flex items-center gap-2 mt-0.5">
          <span className="w-2 h-2 rounded-sm inline-block" style={{ backgroundColor: p.color }} />
          <span className="text-muted">{p.name}:</span>
          <span className="font-mono font-semibold text-foreground">
            {p.dataKey === 'cpo' ? `${fmt(p.value)} ₽` : `${(p.value * 100).toFixed(1)}%`}
          </span>
        </div>
      ))}
    </div>
  )
}

function formatTimeTick(ts: string) {
  const d = new Date(ts)
  if (isNaN(d.getTime())) return ts
  const day = String(d.getDate()).padStart(2, '0')
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const hours = String(d.getHours()).padStart(2, '0')
  const mins = String(d.getMinutes()).padStart(2, '0')
  return `${day}.${month} ${hours}:${mins}`
}

export function MetricsTimeSeriesChart({ data }: Props) {
  if (!data.length) {
    return <div className="text-xs text-muted py-8 text-center">Нет данных</div>
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
        <XAxis
          dataKey="timestamp"
          tick={{ fill: '#7D8590', fontSize: 10 }}
          axisLine={{ stroke: '#30363D' }}
          tickLine={false}
          tickFormatter={formatTimeTick}
          interval="preserveStartEnd"
        />
        <YAxis
          yAxisId="left"
          tick={{ fill: '#7D8590', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={44}
          tickFormatter={v => fmt(v as number)}
          label={{ value: 'CPO ₽', angle: -90, position: 'insideLeft', fill: '#7D8590', fontSize: 10 }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fill: '#7D8590', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={44}
          tickFormatter={v => `${((v as number) * 100).toFixed(0)}%`}
          domain={[0, 1]}
          label={{ value: '%', angle: 90, position: 'insideRight', fill: '#7D8590', fontSize: 10 }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
        />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="cpo"
          name="CPO"
          stroke="#F0883E"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="fill_rate"
          name="Fill Rate"
          stroke="#3FB950"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="wape"
          name="WAPE"
          stroke="#F85149"
          strokeWidth={1.5}
          dot={false}
          strokeDasharray="4 2"
          isAnimationActive={false}
        />
        {data.length > 20 && (
          <Brush
            dataKey="timestamp"
            height={20}
            stroke="#30363D"
            fill="#0D1117"
            tickFormatter={formatTimeTick}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
