import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '../ui/table'
import { cn } from '../../lib/utils'
import type { ApiRouteSummary } from '../../types'
import { useState } from 'react'

interface Props {
  routes: ApiRouteSummary[]
  selectedRouteId: string | null
  onSelectRoute: (id: string) => void
}

type SortKey = keyof ApiRouteSummary
type SortDir = 'asc' | 'desc'

function wapeColor(v: number): string {
  if (v <= 0.15) return 'text-status-green'
  if (v <= 0.30) return 'text-status-yellow'
  return 'text-status-red'
}

function biasColor(v: number): string {
  if (Math.abs(v) <= 5) return 'text-status-green'
  if (Math.abs(v) <= 15) return 'text-status-yellow'
  return 'text-status-red'
}

export function RouteComparisonTable({ routes, selectedRouteId, onSelectRoute }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('route_id')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const sorted = [...routes].sort((a, b) => {
    const av = a[sortKey]
    const bv = b[sortKey]
    if (typeof av === 'string' && typeof bv === 'string') {
      return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
    }
    const diff = (av as number) - (bv as number)
    return sortDir === 'asc' ? diff : -diff
  })

  const arrow = (key: SortKey) => {
    if (sortKey !== key) return ''
    return sortDir === 'asc' ? ' ↑' : ' ↓'
  }

  const columns: { key: SortKey; label: string; format: (v: ApiRouteSummary) => React.ReactNode }[] = [
    { key: 'route_id', label: 'Маршрут', format: r => r.route_id },
    { key: 'avg_predicted', label: 'Ср. прогноз', format: r => r.avg_predicted.toFixed(1) },
    { key: 'avg_actual', label: 'Ср. факт', format: r => r.avg_actual.toFixed(1) },
    {
      key: 'wape',
      label: 'WAPE',
      format: r => (
        <span className={cn('font-mono', wapeColor(r.wape))}>
          {(r.wape * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      key: 'bias',
      label: 'Bias',
      format: r => (
        <span className={cn('font-mono', biasColor(r.bias))}>
          {r.bias > 0 ? '+' : ''}{r.bias.toFixed(1)}
        </span>
      ),
    },
    {
      key: 'fill_rate',
      label: 'Fill Rate',
      format: r => `${(r.fill_rate * 100).toFixed(1)}%`,
    },
    {
      key: 'cpo',
      label: 'CPO ₽',
      format: r => r.cpo.toLocaleString('ru-RU', { maximumFractionDigits: 0 }),
    },
  ]

  return (
    <Table>
      <TableHeader>
        <TableRow>
          {columns.map(col => (
            <TableHead
              key={col.key}
              className="cursor-pointer select-none hover:text-foreground"
              onClick={() => handleSort(col.key)}
            >
              {col.label}{arrow(col.key)}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map(r => (
          <TableRow
            key={r.route_id}
            selected={r.route_id === selectedRouteId}
            onClick={() => onSelectRoute(r.route_id)}
          >
            {columns.map(col => (
              <TableCell key={col.key}>{col.format(r)}</TableCell>
            ))}
          </TableRow>
        ))}
        {sorted.length === 0 && (
          <TableRow>
            <TableCell colSpan={columns.length} className="text-center text-muted py-6">
              Нет данных
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  )
}
