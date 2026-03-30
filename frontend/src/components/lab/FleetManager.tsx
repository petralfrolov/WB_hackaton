import { useState, useCallback, useEffect } from 'react'
import type { Warehouse } from '../../types'
import type { ApiIncomingVehicle } from '../../types'
import { Input } from '../ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table'
import { Trash2, Plus, Clock, RefreshCw } from 'lucide-react'
import { fmt } from '../../lib/utils'
import { cn } from '../../lib/utils'
import { getIncomingVehicles, putIncomingVehicles } from '../../api'
import { useSimulationContext } from '../../context/SimulationContext'

interface FleetManagerProps {
  warehouses: Warehouse[]
}

const HORIZON_LABELS = ['Сейчас (A)', '+2ч (B)', '+4ч (C)', '+6ч (D)'] as const

export function FleetManager({ warehouses }: FleetManagerProps) {
  const { incomingVehicles, setIncomingVehicles } = useSimulationContext()
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string>(warehouses[0]?.id ?? '')
  const [incomingSaving, setIncomingSaving] = useState(false)
  const [incomingError, setIncomingError] = useState<string | null>(null)

  // Load incoming vehicles from API on mount
  useEffect(() => {
    getIncomingVehicles()
      .then(res => setIncomingVehicles(res.incoming))
      .catch(() => { /* silently fall back to context state */ })
  }, [])

  const filtered = warehouses.filter(
    w =>
      w.name.toLowerCase().includes(search.toLowerCase()) ||
      w.city.toLowerCase().includes(search.toLowerCase()),
  )

  const vehicles = warehouses.find(w => w.id === selectedId)?.vehicles ?? []

  const saveIncoming = useCallback(async () => {
    setIncomingSaving(true)
    setIncomingError(null)
    try {
      const res = await putIncomingVehicles(incomingVehicles)
      setIncomingVehicles(res.incoming)
    } catch (err) {
      setIncomingError(err instanceof Error ? err.message : String(err))
    } finally {
      setIncomingSaving(false)
    }
  }, [incomingVehicles])

  const addIncomingRow = useCallback(() => {
    setIncomingVehicles([
      ...incomingVehicles,
      { horizon_idx: 1, vehicle_type: '', count: 1 },
    ])
  }, [incomingVehicles])

  const removeIncomingRow = useCallback((idx: number) => {
    setIncomingVehicles(incomingVehicles.filter((_, i) => i !== idx))
  }, [incomingVehicles])

  const updateIncomingRow = useCallback((idx: number, patch: Partial<ApiIncomingVehicle>) => {
    setIncomingVehicles(
      incomingVehicles.map((row, i) => i === idx ? { ...row, ...patch } : row),
    )
  }, [incomingVehicles])

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
        <div className="section-label">
          {warehouses.find(w => w.id === selectedId)?.name ?? '—'} · Парк ТС
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
                  </TableRow>
                ))}

                {vehicles.length === 0 && (
                  <TableRow>
                    <TableCell className="text-center text-muted py-6" colSpan={4}>
                      Нет ТС на складе.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </div>

        {/* ── Прибывающие ТС (редактор) ─────────────────────────────── */}
        <div>
          <div className="section-label mb-1.5 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-status-yellow" />
              Прибывающие ТС
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={addIncomingRow}
                className="flex items-center gap-1 text-xs text-accent hover:text-accent/80 transition-colors"
              >
                <Plus className="w-3 h-3" />
                Добавить
              </button>
              <button
                onClick={saveIncoming}
                disabled={incomingSaving}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-accent text-white hover:bg-accent/90 disabled:opacity-50 transition-colors"
              >
                <RefreshCw className={cn('w-3 h-3', incomingSaving && 'animate-spin')} />
                {incomingSaving ? 'Сохранение…' : 'Сохранить'}
              </button>
            </div>
          </div>
          {incomingError && (
            <p className="text-xs text-status-red mb-1">{incomingError}</p>
          )}
          <div className="bg-surface border border-border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Тип ТС</TableHead>
                  <TableHead>Горизонт</TableHead>
                  <TableHead className="text-right">Кол-во</TableHead>
                  <TableHead className="text-center w-10">—</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {incomingVehicles.map((row, idx) => (
                  <TableRow key={idx}>
                    <TableCell>
                      <Input
                        placeholder="gazelle_s"
                        value={row.vehicle_type}
                        onChange={e => updateIncomingRow(idx, { vehicle_type: e.target.value })}
                      />
                    </TableCell>
                    <TableCell>
                      <select
                        value={row.horizon_idx}
                        onChange={e => updateIncomingRow(idx, { horizon_idx: Number(e.target.value) as 0|1|2|3 })}
                        className="w-full bg-surface border border-border rounded px-2 py-1 text-sm text-foreground"
                      >
                        {HORIZON_LABELS.map((label, hi) => (
                          <option key={hi} value={hi}>{label}</option>
                        ))}
                      </select>
                    </TableCell>
                    <TableCell>
                      <Input
                        type="number"
                        min="1"
                        value={row.count}
                        onChange={e => updateIncomingRow(idx, { count: Math.max(1, parseInt(e.target.value) || 1) })}
                        className="text-right"
                      />
                    </TableCell>
                    <TableCell className="text-center">
                      <button
                        onClick={() => removeIncomingRow(idx)}
                        className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </TableCell>
                  </TableRow>
                ))}
                {incomingVehicles.length === 0 && (
                  <TableRow>
                    <TableCell className="text-center text-muted py-4" colSpan={4}>
                      Нет прибывающих ТС. Нажмите «Добавить».
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </div>
      </div>
    </div>
  )
}
