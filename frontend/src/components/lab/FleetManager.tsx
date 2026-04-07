import { useEffect, useMemo, useState } from 'react'
import type { Warehouse, ApiIncomingVehicle } from '../../types'
import { Input } from '../ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table'
import { cn, makeHorizonLabels, horizonDisplayLabel } from '../../lib/utils'
import {
  listVehicles,
  updateVehicle,
  listIncoming,
  putIncomingVehicles,
  syncVehicleAcrossWarehouses,
} from '../../api'
import { RefreshCw } from 'lucide-react'

interface FleetManagerProps {
  warehouses: Warehouse[]
}

export function FleetManager({ warehouses }: FleetManagerProps) {
  // Fleet table always uses 30-min granularity
  const horizonLabels = useMemo(() => makeHorizonLabels(0.5), [])
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string>(warehouses[0]?.id ?? '')

  const [vehicles, setVehicles] = useState<any[]>([])
  const [incoming, setIncoming] = useState<ApiIncomingVehicle[]>([])

  const [draftFleet, setDraftFleet] = useState<Record<string, string>>({})
  const [savingFleet, setSavingFleet] = useState<Record<string, boolean>>({})
  const [syncingType, setSyncingType] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Load data for selected warehouse
  useEffect(() => {
    if (!selectedId) return
    const load = async () => {
      try {
        const [vRes, iRes] = await Promise.all([
          listVehicles(selectedId),
          listIncoming(selectedId),
        ])
        setVehicles(vRes)
        setIncoming(iRes.incoming ?? [])
        setDraftFleet({})
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      }
    }
    load()
  }, [selectedId])

  const filteredWarehouses = useMemo(
    () => warehouses.filter(
      w => w.name.toLowerCase().includes(search.toLowerCase()) ||
           w.city.toLowerCase().includes(search.toLowerCase()),
    ),
    [warehouses, search],
  )

  const refreshAll = async () => {
    try {
      const [vRes, iRes] = await Promise.all([
        listVehicles(selectedId),
        listIncoming(selectedId),
      ])
      setVehicles(vRes)
      setIncoming(iRes.incoming ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  // Compute fleet-by-horizon matrix: rows = vehicle types, cols = horizons
  const fleetByHorizon = useMemo(() => {
    return vehicles.map(v => {
      const base = v.available as number
      const additions = Array(horizonLabels.length).fill(0) as number[]
      for (const iv of incoming) {
        if (iv.vehicle_type === v.vehicle_type && iv.horizon_idx < horizonLabels.length) {
          for (let h = iv.horizon_idx; h < horizonLabels.length; h++) {
            additions[h] += iv.count
          }
        }
      }
      return {
        type: v.vehicle_type as string,
        base,
        h: additions.map(a => base + a),
      }
    })
  }, [vehicles, incoming, horizonLabels])

  // Reset drafts when computed data changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { setDraftFleet({}) }, [fleetByHorizon.map(r => r.h.join(',')).join('|')])

  // Handle cell save — mirrors optimizer handleFleetChange logic
  const handleCellBlur = async (vehicleType: string, horizonIdx: number, rowBase: number) => {
    const k = `${vehicleType}__${horizonIdx}`
    const raw = draftFleet[k]
    if (raw === undefined) return

    const row = fleetByHorizon.find(r => r.type === vehicleType)
    if (!row) return

    const currentCount = row.h[horizonIdx]
    const parsed = parseInt(raw, 10)
    const next = Number.isFinite(parsed) ? Math.max(0, parsed) : currentCount
    setDraftFleet(prev => ({ ...prev, [k]: String(next) }))
    if (next === currentCount) return

    setSavingFleet(prev => ({ ...prev, [k]: true }))
    try {
      if (horizonIdx === 0) {
        // Edit base available for this warehouse only
        const v = vehicles.find(vv => vv.vehicle_type === vehicleType)
        if (!v) return
        await updateVehicle(vehicleType, {
          vehicle_type: vehicleType,
          capacity_units: v.capacity_units,
          cost_per_km: v.cost_per_km,
          available: next,
          warehouse_id: selectedId,
          underload_penalty: v.underload_penalty,
          fixed_dispatch_cost: v.fixed_dispatch_cost,
        })
      } else {
        // Edit future horizon: compute new delta and upsert incoming record for this warehouse
        const additionsBelow = incoming
          .filter(iv => iv.vehicle_type === vehicleType && iv.horizon_idx < horizonIdx)
          .reduce((s, iv) => s + iv.count, 0)
        const prevTotal = rowBase + additionsBelow
        const delta = next - prevTotal
        const filtered = incoming.filter(
          iv => !(iv.vehicle_type === vehicleType && iv.horizon_idx === horizonIdx),
        )
        const newList = delta > 0
          ? [...filtered, { vehicle_type: vehicleType, horizon_idx: horizonIdx, count: delta }]
          : filtered
        await putIncomingVehicles(newList, selectedId)
      }
      await refreshAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSavingFleet(prev => ({ ...prev, [k]: false }))
    }
  }

  const selectedWarehouse = warehouses.find(w => w.id === selectedId)

  return (
    <div className="flex gap-4 h-full">
      {/* Warehouse selector */}
      <div className="w-56 shrink-0 flex flex-col gap-2">
        <Input
          placeholder="Поиск склада…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div className="flex-1 overflow-y-auto space-y-0.5 rounded-lg border border-border bg-surface">
          {filteredWarehouses.map(w => (
            <button
              key={w.id}
              onClick={() => setSelectedId(w.id)}
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
          {filteredWarehouses.length === 0 && (
            <p className="px-3 py-4 text-xs text-muted text-center">Ничего не найдено</p>
          )}
        </div>
      </div>

      {/* Single editable fleet-by-horizon table */}
      <div className="flex-1 flex flex-col gap-3 min-w-0">
        {error && (
          <div className="text-xs text-status-red bg-status-red/10 border border-status-red/40 rounded px-3 py-2">
            {error}
          </div>
        )}

        <div className="bg-surface border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 border-b border-border">
            <span className="section-label">
              {selectedWarehouse?.name ?? '—'} · Доступные ТС по горизонтам
            </span>
          </div>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Тип ТС</TableHead>
                  {horizonLabels.map(label => (
                    <TableHead key={label} className="text-right">
                      {horizonDisplayLabel(label)}
                    </TableHead>
                  ))}
                  <TableHead className="text-center w-10">Синхр.</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {fleetByHorizon.map(row => (
                  <TableRow key={row.type}>
                    <TableCell className="font-mono text-sm">{row.type}</TableCell>
                    {row.h.map((count, i) => {
                      const k = `${row.type}__${i}`
                      const draftVal = draftFleet[k] ?? String(count)
                      const isSaving = savingFleet[k]
                      const prev = i > 0 ? row.h[i - 1] : count
                      const isIncrease = count > prev
                      return (
                        <TableCell key={i} className="text-right">
                          <input
                            type="number"
                            min="0"
                            value={draftVal}
                            disabled={isSaving}
                            onChange={e =>
                              setDraftFleet(prev => ({ ...prev, [k]: e.target.value }))
                            }
                            onFocus={() =>
                              setDraftFleet(prev => ({ ...prev, [k]: String(count) }))
                            }
                            onBlur={() => handleCellBlur(row.type, i, row.base)}
                            className={`w-16 h-7 rounded bg-elevated border border-border px-2 text-right text-sm font-mono focus:outline-none focus:border-accent disabled:opacity-50 ${
                              count === 0
                                ? 'text-status-red'
                                : isIncrease
                                  ? 'text-status-green font-semibold'
                                  : 'text-foreground'
                            }`}
                          />
                        </TableCell>
                      )
                    })}
                    <TableCell className="text-center">
                      <button
                        disabled={syncingType === row.type}
                        onClick={async () => {
                          setSyncingType(row.type)
                          try {
                            await syncVehicleAcrossWarehouses(row.type, selectedId)
                          } catch (err) {
                            setError(err instanceof Error ? err.message : String(err))
                          } finally {
                            setSyncingType(null)
                          }
                        }}
                        className="inline-flex items-center justify-center w-7 h-7 rounded hover:bg-accent/10 text-muted hover:text-accent transition-colors disabled:opacity-50"
                        title="Синхронизировать на все склады"
                      >
                        <RefreshCw className={cn('w-3.5 h-3.5', syncingType === row.type && 'animate-spin')} />
                      </button>
                    </TableCell>
                  </TableRow>
                ))}
                {fleetByHorizon.length === 0 && (
                  <TableRow>
                    <TableCell
                      className="text-center text-muted py-6"
                      colSpan={2 + horizonLabels.length}
                    >
                      Нет типов ТС. Создайте их на вкладке «Конструктор ТС».
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </div>

        <p className="text-xs text-muted">
          Столбец «{horizonDisplayLabel(horizonLabels[0])}» — базовое кол-во ТС на складе.
          Остальные — с учётом прибывающих к горизонту.
          Редактирование любого столбца автоматически обновляет данные для выбранного склада.
        </p>
      </div>
    </div>
  )
}
