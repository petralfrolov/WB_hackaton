import { useState, useCallback } from 'react'
import type { Warehouse, VehicleType } from '../../types'
import { Input } from '../ui/input'
import { Button } from '../ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table'
import { Trash2, Plus, Check, X, Clock } from 'lucide-react'
import { fmt } from '../../lib/utils'
import { cn } from '../../lib/utils'

interface FleetManagerProps {
  warehouses: Warehouse[]
}

type WarehouseFleet = Record<string, VehicleType[]>

interface NewVehicleForm {
  name: string
  capacity: string
  costPerKm: string
  available: string
}

const EMPTY_FORM: NewVehicleForm = {
  name: '',
  capacity: '',
  costPerKm: '',
  available: '',
}

export function FleetManager({ warehouses }: FleetManagerProps) {
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string>(warehouses[0]?.id ?? '')
  const [fleet, setFleet] = useState<WarehouseFleet>(() =>
    Object.fromEntries(warehouses.map(w => [w.id, [...w.vehicles]])),
  )
  const [adding, setAdding] = useState(false)
  const [form, setForm] = useState<NewVehicleForm>(EMPTY_FORM)

  const filtered = warehouses.filter(
    w =>
      w.name.toLowerCase().includes(search.toLowerCase()) ||
      w.city.toLowerCase().includes(search.toLowerCase()),
  )

  const vehicles = fleet[selectedId] ?? []

  const deleteVehicle = useCallback((vehicleId: string) => {
    setFleet(prev => ({
      ...prev,
      [selectedId]: (prev[selectedId] ?? []).filter(v => v.id !== vehicleId),
    }))
  }, [selectedId])

  const saveVehicle = useCallback(() => {
    const capacity = parseInt(form.capacity, 10)
    const costPerKm = parseInt(form.costPerKm, 10)
    const available = parseInt(form.available, 10)

    if (!form.name.trim() || isNaN(capacity) || isNaN(costPerKm) || isNaN(available)) return

    const newVehicle: VehicleType = {
      id: `v-custom-${Date.now()}`,
      name: form.name.trim(),
      capacity,
      costPerKm,
      available,
    }

    setFleet(prev => ({
      ...prev,
      [selectedId]: [...(prev[selectedId] ?? []), newVehicle],
    }))
    setAdding(false)
    setForm(EMPTY_FORM)
  }, [form, selectedId])

  return (
    <div className="flex gap-4 h-full">
      {/* Warehouse list */}
      <div className="w-56 shrink-0 flex flex-col gap-2">
        <Input
          placeholder="Поиск склада…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div className="flex-1 overflow-y-auto space-y-0.5 rounded-lg border border-border bg-surface">
          {filtered.map(w => (
            <button
              key={w.id}
              onClick={() => {
                setSelectedId(w.id)
                setAdding(false)
                setForm(EMPTY_FORM)
              }}
              className={cn(
                'w-full text-left px-3 py-2.5 text-sm transition-colors',
                selectedId === w.id
                  ? 'bg-accent/10 text-accent border-l-2 border-accent'
                  : 'text-foreground hover:bg-elevated',
              )}
            >
              <div className="font-medium truncate">{w.name}</div>
              <div className="text-xs text-muted">{w.city}</div>
            </button>
          ))}
          {filtered.length === 0 && (
            <p className="px-3 py-4 text-xs text-muted text-center">Ничего не найдено</p>
          )}
        </div>
      </div>

      {/* Fleet table */}
      <div className="flex-1 flex flex-col gap-3 min-w-0 overflow-y-auto">
        <div className="flex items-center justify-between">
          <div className="section-label">
            {warehouses.find(w => w.id === selectedId)?.name ?? '—'} · Парк ТС
          </div>
          <Button size="sm" variant="outline" onClick={() => { setAdding(v => !v); setForm(EMPTY_FORM) }}>
            <Plus className="w-3.5 h-3.5" />
            Добавить ТС
          </Button>
        </div>

        {/* ── Ожидают на складе ─────────────────────────────────────────── */}
        <div>
          <div className="section-label mb-1.5 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-status-green inline-block" />
            Ожидают на складе
          </div>
          <div className="bg-surface border border-border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Название</TableHead>
                  <TableHead className="text-right">Вместимость</TableHead>
                  <TableHead className="text-right">₽/км</TableHead>
                  <TableHead className="text-right">Доступно</TableHead>
                  <TableHead className="text-center w-12">—</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {vehicles.map(v => (
                  <TableRow key={v.id}>
                    <TableCell>{v.name}</TableCell>
                    <TableCell className="text-right font-mono">{fmt(v.capacity)}</TableCell>
                    <TableCell className="text-right font-mono">{fmt(v.costPerKm)}</TableCell>
                    <TableCell className="text-right font-mono text-status-green font-semibold">
                      {v.available}
                    </TableCell>
                    <TableCell className="text-center">
                      <button
                        onClick={() => deleteVehicle(v.id)}
                        className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors"
                        aria-label="Удалить"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </TableCell>
                  </TableRow>
                ))}

                {/* Add row */}
                {adding && (
                  <TableRow>
                    <TableCell>
                      <Input placeholder="Название" value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))} autoFocus />
                    </TableCell>
                    <TableCell>
                      <Input placeholder="0" type="number" min="1" value={form.capacity} onChange={e => setForm(p => ({ ...p, capacity: e.target.value }))} className="text-right" />
                    </TableCell>
                    <TableCell>
                      <Input placeholder="0" type="number" min="1" value={form.costPerKm} onChange={e => setForm(p => ({ ...p, costPerKm: e.target.value }))} className="text-right" />
                    </TableCell>
                    <TableCell>
                      <Input placeholder="0" type="number" min="0" value={form.available} onChange={e => setForm(p => ({ ...p, available: e.target.value }))} className="text-right" />
                    </TableCell>
                    <TableCell className="text-center">
                      <div className="flex gap-1 justify-center">
                        <button onClick={saveVehicle} className="p-1 rounded hover:bg-status-green/15 text-muted hover:text-status-green transition-colors" aria-label="Сохранить">
                          <Check className="w-4 h-4" />
                        </button>
                        <button onClick={() => { setAdding(false); setForm(EMPTY_FORM) }} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors" aria-label="Отмена">
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </TableCell>
                  </TableRow>
                )}

                {vehicles.length === 0 && !adding && (
                  <TableRow>
                    <TableCell className="text-center text-muted py-6" colSpan={5}>
                      Нет ТС на складе. Нажмите «+ Добавить ТС».
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </div>

        {/* ── Скоро прибудут ────────────────────────────────────────────── */}
        {(() => {
          const incoming = vehicles.flatMap(v =>
            (v.incoming ?? []).map(inc => ({
              vehicleName: v.name,
              capacity: v.capacity,
              costPerKm: v.costPerKm,
              ...inc,
            })),
          ).filter(inc => inc.count > 0)

          if (incoming.length === 0) return null

          return (
            <div>
              <div className="section-label mb-1.5 flex items-center gap-1.5">
                <Clock className="w-3 h-3 text-status-yellow" />
                Скоро прибудут
              </div>
              <div className="bg-surface border border-border rounded-lg overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>ТС</TableHead>
                      <TableHead className="text-right">Вместимость</TableHead>
                      <TableHead className="text-right">₽/км</TableHead>
                      <TableHead className="text-right">Кол-во</TableHead>
                      <TableHead className="text-right">Прибытие</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {incoming.map(inc => (
                      <TableRow key={inc.id}>
                        <TableCell>{inc.vehicleName}</TableCell>
                        <TableCell className="text-right font-mono">{fmt(inc.capacity)}</TableCell>
                        <TableCell className="text-right font-mono">{fmt(inc.costPerKm)}</TableCell>
                        <TableCell className="text-right font-mono font-semibold text-status-yellow">
                          {inc.count}
                        </TableCell>
                        <TableCell className="text-right">
                          <span className="flex items-center justify-end gap-1 text-status-yellow font-mono text-sm">
                            <Clock className="w-3 h-3" />
                            ~{inc.arrivalMinutes} мин
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )
        })()}
      </div>
    </div>
  )
}
