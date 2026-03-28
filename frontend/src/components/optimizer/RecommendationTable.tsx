import { useState, useMemo, useEffect } from 'react'
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

type SortField = 'from' | 'to' | 'forecast' | 'status'
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
  warehouseLabelById?: Record<string, string>
}

function splitRoute(route: string): { from: string; to: string } {
  const parts = route.split('→').map(s => s.trim())
  return {
    from: parts[0] ?? route,
    to: parts[1] ?? '—',
  }
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
  warehouseLabelById,
}: RecommendationTableProps) {
  const [filterFromId, setFilterFromId] = useState(warehouseFilter ?? '')
  const [filterStatus, setFilterStatus] = useState<'' | TransportRecommendation['status']>('')
  const [sortField, setSortField] = useState<SortField | null>(null)
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  useEffect(() => {
    setFilterFromId(warehouseFilter ?? '')
  }, [warehouseFilter])

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
    if (filterFromId) {
      result = result.filter(r => r.warehouseId === filterFromId)
    }
    if (filterStatus) {
      result = result.filter(r => r.status === filterStatus)
    }
    // Sort
    result = [...result].sort((a, b) => {
      if (!sortField) {
        return STATUS_ORDER[a.status] - STATUS_ORDER[b.status]
      }
      const aRoute = splitRoute(a.route)
      const bRoute = splitRoute(b.route)
      const aFrom = warehouseLabelById?.[a.warehouseId] ?? aRoute.from
      const bFrom = warehouseLabelById?.[b.warehouseId] ?? bRoute.from

      let cmp = 0
      if (sortField === 'from') cmp = aFrom.localeCompare(bFrom)
      else if (sortField === 'to') cmp = aRoute.to.localeCompare(bRoute.to)
      else if (sortField === 'forecast') cmp = a.forecast - b.forecast
      else if (sortField === 'status') cmp = STATUS_ORDER[a.status] - STATUS_ORDER[b.status]
      return sortDir === 'asc' ? cmp : -cmp
    })
    return result
  }, [recommendations, filterFromId, filterStatus, sortField, sortDir, warehouseLabelById])

  const fromOptions = useMemo(() => {
    return Array.from(new Set(recommendations.map(r => r.warehouseId)))
      .map(id => ({ id, label: warehouseLabelById?.[id] ?? splitRoute(recommendations.find(r => r.warehouseId === id)?.route ?? id).from }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [recommendations, warehouseLabelById])

  return (
    <div className="bg-surface rounded-lg border border-border overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between gap-3">
        <span className="section-label">Рекомендации оптимизатора</span>
        {filterFromId && (
          <span className="text-xs text-accent font-mono bg-accent/10 rounded px-2 py-0.5">
            Откуда: {warehouseLabelById?.[filterFromId] ?? filterFromId}
          </span>
        )}
      </div>

      {/* Filter row */}
      <div className="px-4 py-2 border-b border-border flex items-center gap-3 bg-elevated/40">
        <select
          value={filterFromId}
          onChange={e => setFilterFromId(e.target.value)}
          className="flex-1 text-xs bg-elevated border border-border rounded px-2 py-1.5 text-foreground focus:outline-none focus:border-accent"
        >
          <option value="">Откуда: все склады</option>
          {fromOptions.map(o => (
            <option key={o.id} value={o.id}>{o.label}</option>
          ))}
        </select>
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
        {(filterFromId || filterStatus) && (
          <button
            onClick={() => { setFilterFromId(''); setFilterStatus('') }}
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
                onClick={() => handleSortClick('from')}
                className="flex items-center text-left hover:text-foreground transition-colors"
              >
                Откуда <SortIcon field="from" sortField={sortField} sortDir={sortDir} />
              </button>
            </TableHead>
            <TableHead>
              <button
                onClick={() => handleSortClick('to')}
                className="flex items-center text-left hover:text-foreground transition-colors"
              >
                Куда <SortIcon field="to" sortField={sortField} sortDir={sortDir} />
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
              <TableCell colSpan={6} className="text-center text-muted py-8 text-sm">
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
                  <span className="font-medium text-foreground">
                    {warehouseLabelById?.[rec.warehouseId] ?? splitRoute(rec.route).from}
                  </span>
                </TableCell>
                <TableCell>
                  <span className="text-muted">{splitRoute(rec.route).to}</span>
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
