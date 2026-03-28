import { useState, useMemo } from 'react'
import type { TransportRecommendation } from '../../types'
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '../ui/table'
import { Badge } from '../ui/badge'
import type { BadgeVariant } from '../ui/badge'
import { Button } from '../ui/button'
import { fmt } from '../../lib/utils'
import { CheckCircle2, ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react'

type SortField = 'route' | 'forecast' | 'status'
type SortDir = 'asc' | 'desc'

const STATUS_BADGE: Record<TransportRecommendation['status'], BadgeVariant> = {
  pending: 'pending',
  called: 'called',
  warning: 'warning',
}

const STATUS_LABEL: Record<TransportRecommendation['status'], string> = {
  pending: 'Ожидает',
  called: 'Вызван',
  warning: 'Внимание',
}

const STATUS_ORDER: Record<TransportRecommendation['status'], number> = {
  warning: 0,
  pending: 1,
  called: 2,
}

interface RecommendationTableProps {
  recommendations: TransportRecommendation[]
  selectedId: string | null
  onSelect: (rec: TransportRecommendation) => void
  onCall: (id: string) => void
  warehouseFilter?: string
}

function SortIcon({ field, sortField, sortDir }: { field: SortField; sortField: SortField | null; sortDir: SortDir }) {
  if (sortField !== field) return <ChevronsUpDown className="w-3 h-3 ml-1 opacity-30" />
  return sortDir === 'asc'
    ? <ChevronUp className="w-3 h-3 ml-1 text-accent" />
    : <ChevronDown className="w-3 h-3 ml-1 text-accent" />
}

export function RecommendationTable({
  recommendations,
  selectedId,
  onSelect,
  onCall,
  warehouseFilter,
}: RecommendationTableProps) {
  const [filterRoute, setFilterRoute] = useState('')
  const [filterStatus, setFilterStatus] = useState<'' | TransportRecommendation['status']>('')
  const [sortField, setSortField] = useState<SortField | null>(null)
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const handleSortClick = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('asc')
    }
  }

  const filtered = useMemo(() => {
    let result = recommendations
    if (warehouseFilter) {
      result = result.filter(r => r.warehouseId === warehouseFilter)
    }
    if (filterRoute.trim()) {
      const q = filterRoute.trim().toLowerCase()
      result = result.filter(r => r.route.toLowerCase().includes(q))
    }
    if (filterStatus) {
      result = result.filter(r => r.status === filterStatus)
    }
    // Sort
    result = [...result].sort((a, b) => {
      if (!sortField) {
        return STATUS_ORDER[a.status] - STATUS_ORDER[b.status]
      }
      let cmp = 0
      if (sortField === 'route') cmp = a.route.localeCompare(b.route)
      else if (sortField === 'forecast') cmp = a.forecast - b.forecast
      else if (sortField === 'status') cmp = STATUS_ORDER[a.status] - STATUS_ORDER[b.status]
      return sortDir === 'asc' ? cmp : -cmp
    })
    return result
  }, [recommendations, warehouseFilter, filterRoute, filterStatus, sortField, sortDir])

  return (
    <div className="bg-surface rounded-lg border border-border overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between gap-3">
        <span className="section-label">Рекомендации оптимизатора</span>
        {warehouseFilter && (
          <span className="text-xs text-accent font-mono bg-accent/10 rounded px-2 py-0.5">
            Склад: {warehouseFilter}
          </span>
        )}
      </div>

      {/* Filter row */}
      <div className="px-4 py-2 border-b border-border flex items-center gap-3 bg-elevated/40">
        <input
          type="text"
          placeholder="Фильтр по маршруту…"
          value={filterRoute}
          onChange={e => setFilterRoute(e.target.value)}
          className="flex-1 text-xs bg-elevated border border-border rounded px-2 py-1.5 text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
        />
        <select
          value={filterStatus}
          onChange={e => setFilterStatus(e.target.value as typeof filterStatus)}
          className="text-xs bg-elevated border border-border rounded px-2 py-1.5 text-foreground focus:outline-none focus:border-accent"
        >
          <option value="">Все статусы</option>
          <option value="warning">Внимание</option>
          <option value="pending">Ожидает</option>
          <option value="called">Вызван</option>
        </select>
        {(filterRoute || filterStatus) && (
          <button
            onClick={() => { setFilterRoute(''); setFilterStatus('') }}
            className="text-xs text-muted hover:text-foreground transition-colors px-1"
          >
            Сбросить
          </button>
        )}
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>
              <button
                onClick={() => handleSortClick('route')}
                className="flex items-center text-left hover:text-foreground transition-colors"
              >
                Маршрут <SortIcon field="route" sortField={sortField} sortDir={sortDir} />
              </button>
            </TableHead>
            <TableHead className="text-right">
              <button
                onClick={() => handleSortClick('forecast')}
                className="flex items-center ml-auto hover:text-foreground transition-colors"
              >
                Прогноз <SortIcon field="forecast" sortField={sortField} sortDir={sortDir} />
              </button>
            </TableHead>
            <TableHead>Рекомендация ТС</TableHead>
            <TableHead>
              <button
                onClick={() => handleSortClick('status')}
                className="flex items-center hover:text-foreground transition-colors"
              >
                Статус <SortIcon field="status" sortField={sortField} sortDir={sortDir} />
              </button>
            </TableHead>
            <TableHead className="text-center">Действие</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filtered.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center text-muted py-8 text-sm">
                Нет данных по выбранным фильтрам
              </TableCell>
            </TableRow>
          ) : (
            filtered.map(rec => (
              <TableRow
                key={rec.id}
                onClick={() => onSelect(rec)}
                selected={rec.id === selectedId}
              >
                <TableCell>
                  <span className="font-medium text-foreground">{rec.route}</span>
                </TableCell>
                <TableCell className="text-right">
                  <span className="font-mono">{fmt(rec.forecast)}</span>
                </TableCell>
                <TableCell>
                  <span className="text-muted">{rec.recommendation}</span>
                </TableCell>
                <TableCell>
                  <Badge variant={STATUS_BADGE[rec.status]}>
                    {STATUS_LABEL[rec.status]}
                  </Badge>
                </TableCell>
                <TableCell className="text-center">
                  {rec.status === 'pending' ? (
                    <Button
                      size="sm"
                      onClick={e => {
                        e.stopPropagation()
                        onCall(rec.id)
                      }}
                    >
                      Вызвать
                    </Button>
                  ) : rec.status === 'called' ? (
                    <Button size="sm" variant="success" disabled>
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Вызван
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={e => {
                        e.stopPropagation()
                        onCall(rec.id)
                      }}
                    >
                      Вызвать
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  )
}
