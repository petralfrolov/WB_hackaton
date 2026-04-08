import { cn } from '../../lib/utils'
import type { ApiAggregateMetrics } from '../../types'

interface Props {
  aggregate: ApiAggregateMetrics
}

function kpiColor(value: number, thresholds: [number, number], invert = false): string {
  const [good, warn] = thresholds
  if (invert) {
    if (value <= good) return 'text-status-green'
    if (value <= warn) return 'text-status-yellow'
    return 'text-status-red'
  }
  if (value >= good) return 'text-status-green'
  if (value >= warn) return 'text-status-yellow'
  return 'text-status-red'
}

export function KpiCards({ aggregate }: Props) {
  const cards = [
    {
      label: 'Средний CPO',
      value: `${aggregate.avg_cpo.toLocaleString('ru-RU', { maximumFractionDigits: 0 })} ₽`,
      sub: 'стоимость на единицу',
      color: 'text-foreground',
    },
    {
      label: 'Fill Rate',
      value: `${(aggregate.avg_fill_rate * 100).toFixed(1)}%`,
      sub: 'заполненность ТС',
      color: kpiColor(aggregate.avg_fill_rate, [0.7, 0.5]),
    },
    {
      label: 'WAPE прогноза',
      value: `${(aggregate.wape * 100).toFixed(1)}%`,
      sub: 'взвешенная ошибка',
      color: kpiColor(aggregate.wape, [0.15, 0.30], true),
    },
    {
      label: 'Покрытие ДИ',
      value: `${(aggregate.realized_coverage * 100).toFixed(1)}%`,
      sub: 'факт в интервале',
      color: kpiColor(aggregate.realized_coverage, [0.85, 0.75]),
    },
  ]

  return (
    <div className="grid grid-cols-4 gap-3">
      {cards.map(c => (
        <div
          key={c.label}
          className="rounded-lg bg-surface border border-border px-4 py-3 flex flex-col"
        >
          <span className="text-[10px] text-muted uppercase tracking-widest">{c.label}</span>
          <span className={cn('text-2xl font-bold font-mono mt-1', c.color)}>{c.value}</span>
          <span className="text-[10px] text-muted mt-0.5">{c.sub}</span>
        </div>
      ))}
    </div>
  )
}
