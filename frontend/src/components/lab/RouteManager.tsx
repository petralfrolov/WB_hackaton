import { useState } from 'react'
import type { RouteDistance, Warehouse } from '../../types'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Trash2, Plus, Check, X, RouteIcon } from 'lucide-react'
import { fmt } from '../../lib/utils'

interface RouteManagerProps {
  warehouses: Warehouse[]
  initialRoutes: RouteDistance[]
}

export function RouteManager({ warehouses, initialRoutes }: RouteManagerProps) {
  const [routes, setRoutes] = useState<RouteDistance[]>(initialRoutes)
  const [adding, setAdding] = useState(false)
  const [form, setForm] = useState({ fromId: '', toId: '', distanceKm: '' })

  const warehouseOptions = warehouses.map(w => ({ id: w.id, label: `${w.city} — ${w.name}` }))

  const saveRoute = () => {
    const dist = parseInt(form.distanceKm, 10)
    if (!form.fromId || !form.toId || form.fromId === form.toId || isNaN(dist) || dist <= 0) return

    const from = warehouses.find(w => w.id === form.fromId)
    const to = warehouses.find(w => w.id === form.toId)
    if (!from || !to) return

    const newRoute: RouteDistance = {
      id: `rd-${Date.now()}`,
      fromId: form.fromId,
      toId: form.toId,
      fromCity: from.city,
      toCity: to.city,
      distanceKm: dist,
    }
    setRoutes(prev => [...prev, newRoute])
    setAdding(false)
    setForm({ fromId: '', toId: '', distanceKm: '' })
  }

  const deleteRoute = (id: string) => {
    setRoutes(prev => prev.filter(r => r.id !== id))
  }

  return (
    <div className="flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <RouteIcon className="w-4 h-4 text-muted" />
          <span className="section-label">Маршруты и расстояния</span>
        </div>
        <Button size="sm" variant="outline" onClick={() => { setAdding(v => !v); setForm({ fromId: '', toId: '', distanceKm: '' }) }}>
          <Plus className="w-3.5 h-3.5" />
          Добавить маршрут
        </Button>
      </div>

      {/* Add form */}
      {adding && (
        <div className="bg-surface border border-accent/30 rounded-lg p-4 flex flex-col gap-3">
          <div className="section-label">Новый маршрут</div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-[11px] text-muted mb-1 block">Откуда (склад)</label>
              <select
                className="w-full h-8 rounded bg-elevated border border-border px-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
                value={form.fromId}
                onChange={e => setForm(p => ({ ...p, fromId: e.target.value }))}
              >
                <option value="">Выберите склад...</option>
                {warehouseOptions.map(o => (
                  <option key={o.id} value={o.id}>{o.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[11px] text-muted mb-1 block">Куда (склад)</label>
              <select
                className="w-full h-8 rounded bg-elevated border border-border px-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
                value={form.toId}
                onChange={e => setForm(p => ({ ...p, toId: e.target.value }))}
              >
                <option value="">Выберите склад...</option>
                {warehouseOptions.filter(o => o.id !== form.fromId).map(o => (
                  <option key={o.id} value={o.id}>{o.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[11px] text-muted mb-1 block">Расстояние (км)</label>
              <Input
                type="number"
                min="1"
                placeholder="0"
                value={form.distanceKm}
                onChange={e => setForm(p => ({ ...p, distanceKm: e.target.value }))}
              />
            </div>
          </div>
          <div className="flex gap-2 justify-end">
            <Button size="sm" variant="outline" onClick={() => setAdding(false)}>
              <X className="w-3.5 h-3.5" />
              Отмена
            </Button>
            <Button size="sm" onClick={saveRoute}>
              <Check className="w-3.5 h-3.5" />
              Сохранить
            </Button>
          </div>
        </div>
      )}

      {/* Routes table */}
      <div className="bg-surface border border-border rounded-lg overflow-hidden flex-1">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Откуда</TableHead>
              <TableHead>Куда</TableHead>
              <TableHead className="text-right">Расстояние</TableHead>
              <TableHead className="text-center w-12">—</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {routes.map(r => (
              <TableRow key={r.id}>
                <TableCell>
                  <span className="font-medium text-foreground">{r.fromCity}</span>
                  <span className="text-xs text-muted block">
                    {warehouses.find(w => w.id === r.fromId)?.name}
                  </span>
                </TableCell>
                <TableCell>
                  <span className="font-medium text-foreground">{r.toCity}</span>
                  <span className="text-xs text-muted block">
                    {warehouses.find(w => w.id === r.toId)?.name}
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  <span className="font-mono text-accent font-semibold">{fmt(r.distanceKm)}</span>
                  <span className="text-muted text-xs ml-1">км</span>
                </TableCell>
                <TableCell className="text-center">
                  <button
                    onClick={() => deleteRoute(r.id)}
                    className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors"
                    aria-label="Удалить маршрут"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </TableCell>
              </TableRow>
            ))}
            {routes.length === 0 && (
              <TableRow>
                <TableCell className="text-center text-muted py-8" colSpan={4}>
                  Нет маршрутов. Нажмите «+ Добавить маршрут».
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
