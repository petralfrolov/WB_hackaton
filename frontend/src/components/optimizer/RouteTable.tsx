import { AlertTriangle, Loader2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Warehouse, RouteDistance, ApiDispatchResponse, VehicleType, ApiIncomingVehicle } from '../../types'
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '../ui/table'
import { fmt } from '../../lib/utils'

const HORIZON_LABELS = ['Сейчас', '+2ч', '+4ч', '+6ч'] as const

function computeFleetByHorizon(
  vehicleTypes: VehicleType[],
  incomingVehicles: ApiIncomingVehicle[],
): Array<{ type: string; h: [number, number, number, number] }> {
  return vehicleTypes.map(vt => {
    const base = vt.available
    const additions: [number, number, number, number] = [0, 0, 0, 0]
    for (const iv of incomingVehicles) {
      if (iv.vehicle_type === vt.id) {
        for (let h = iv.horizon_idx; h < 4; h++) {
          additions[h] += iv.count
        }
      }
    }
    return {
      type: vt.id,
      h: [base + additions[0], base + additions[1], base + additions[2], base + additions[3]],
    }
  })
}

interface RouteTableProps {
  warehouse: Warehouse | null
  warehouseRoutes: RouteDistance[]
  dispatchResult: ApiDispatchResponse | null
  loading: boolean
  error: string | null
  selectedRouteId: string | null
  onSelectRoute: (routeId: string) => void
  onChangeReadyToShip: (routeId: string, value: number) => void
  onCallRoute: (routeId: string) => Promise<string>
  onCallAllRoutes?: (routeIds: string[]) => Promise<void>
  onReadyDirtyChange?: (dirty: boolean) => void
  vehicleTypes?: VehicleType[]
  incomingVehicles?: ApiIncomingVehicle[]
  onFleetChange?: (vehicleType: string, horizonIdx: 0 | 1 | 2 | 3, newCount: number) => Promise<void>
  analysisDateTime?: string
}

function forecastColor(demand: number, leftover: number | null, vehiclesCount: number | null): string {
  if (demand <= 0) return 'text-muted'
  if (leftover === null) return 'text-foreground' // no dispatch result yet
  if (vehiclesCount === 0) return 'text-status-red font-semibold'   // nothing dispatched
  if (leftover >= 1) return 'text-status-yellow font-semibold'       // partial coverage
  return 'text-foreground'                                           // fully covered
}

interface RouteRow {
  routeId: string
  fromCity: string
  toCity: string
  distanceKm: number
  readyToShip: number
  h0: number | null
  h1: number | null
  h2: number | null
  h0Lo: number | null
  h1Lo: number | null
  h2Lo: number | null
  h0Hi: number | null
  h1Hi: number | null
  h2Hi: number | null
  h0Leftover: number | null
  h1Leftover: number | null
  h2Leftover: number | null
  h0Vehicles: number | null
  h1Vehicles: number | null
  h2Vehicles: number | null
}

export function RouteTable({
  warehouse,
  warehouseRoutes,
  dispatchResult,
  loading,
  error,
  selectedRouteId,
  onSelectRoute,
  onChangeReadyToShip,
  onCallRoute,
  onCallAllRoutes,
  onReadyDirtyChange,
  vehicleTypes = [],
  incomingVehicles = [],
  onFleetChange,
  analysisDateTime = '',
}: RouteTableProps) {
  const calledLsKey = (routeId: string) => `called_${routeId}__${analysisDateTime}`
  const [draftReady, setDraftReady] = useState<Record<string, string>>({})
  const [draftFleet, setDraftFleet] = useState<Record<string, string>>({})
  const [savingFleet, setSavingFleet] = useState<Record<string, boolean>>({})
  const [called, setCalled] = useState<Record<string, boolean>>({})
  const [bulkCalled, setBulkCalled] = useState(false)

  useEffect(() => {
    setDraftReady(prev => {
      const next = { ...prev }
      let changed = false
      for (const route of warehouseRoutes) {
        if (next[route.id] === undefined) {
          next[route.id] = String(route.readyToShip)
          changed = true
        }
      }
      return changed ? next : prev
    })
    // сохраняем статусы вызова в localStorage, ключ — route_id + analysisDateTime
    const calledFromLS: Record<string, boolean> = {}
    for (const route of warehouseRoutes) {
      const val = localStorage.getItem(calledLsKey(route.id))
      calledFromLS[route.id] = val === '1'
    }
    setCalled(calledFromLS)
    setBulkCalled(warehouseRoutes.length > 0 && warehouseRoutes.every(r => calledFromLS[r.id]))
  }, [warehouseRoutes, analysisDateTime]) // eslint-disable-line react-hooks/exhaustive-deps

  // Build lookup: route_id → forecast + coverage by horizon
  const forecastMap = new Map<string, { h0: number; h1: number; h2: number; h0Lo: number; h1Lo: number; h2Lo: number; h0Hi: number; h1Hi: number; h2Hi: number; h0l: number; h1l: number; h2l: number; h0v: number; h1v: number; h2v: number }>()
  if (dispatchResult) {
    for (const rp of dispatchResult.routes) {
      const hB = rp.plan.find(r => r.horizon === 'B: +2h')
      const hC = rp.plan.find(r => r.horizon === 'C: +4h')
      const hD = rp.plan.find(r => r.horizon === 'D: +6h')
      forecastMap.set(rp.route_id, {
        h0: hB?.demand_new ?? 0,
        h1: hC?.demand_new ?? 0,
        h2: hD?.demand_new ?? 0,
        h0Lo: hB?.demand_lower ?? 0,
        h1Lo: hC?.demand_lower ?? 0,
        h2Lo: hD?.demand_lower ?? 0,
        h0Hi: hB?.demand_upper ?? 0,
        h1Hi: hC?.demand_upper ?? 0,
        h2Hi: hD?.demand_upper ?? 0,
        h0l: hB?.leftover_stock ?? 0,
        h1l: hC?.leftover_stock ?? 0,
        h2l: hD?.leftover_stock ?? 0,
        h0v: hB?.vehicles_count ?? 0,
        h1v: hC?.vehicles_count ?? 0,
        h2v: hD?.vehicles_count ?? 0,
      })
    }
  }

  const rows: RouteRow[] = warehouseRoutes.map(r => {
    const fd = forecastMap.get(r.id)
    return {
      routeId: r.id,
      fromCity: r.fromCity,
      toCity: r.toCity,
      distanceKm: r.distanceKm,
      readyToShip: r.readyToShip,
      h0: fd?.h0 ?? null,
      h1: fd?.h1 ?? null,
      h2: fd?.h2 ?? null,
      h0Lo: fd?.h0Lo ?? null,
      h1Lo: fd?.h1Lo ?? null,
      h2Lo: fd?.h2Lo ?? null,
      h0Hi: fd?.h0Hi ?? null,
      h1Hi: fd?.h1Hi ?? null,
      h2Hi: fd?.h2Hi ?? null,
      h0Leftover: fd != null ? fd.h0l : null,
      h1Leftover: fd != null ? fd.h1l : null,
      h2Leftover: fd != null ? fd.h2l : null,
      h0Vehicles: fd != null ? fd.h0v : null,
      h1Vehicles: fd != null ? fd.h1v : null,
      h2Vehicles: fd != null ? fd.h2v : null,
    }
  })

  const hasShortfall = dispatchResult !== null && dispatchResult.routes.some(r => r.coverage_min < -0.5)

  const fleetByHorizon = computeFleetByHorizon(vehicleTypes, incomingVehicles)

  const handleCall = async (routeId: string) => {
    const already = called[routeId]
    const nextState = !already
    try {
      if (!already) {
        const json = await onCallRoute(routeId)
        const blob = new Blob([json], { type: 'application/json;charset=utf-8' })
        const url = URL.createObjectURL(blob)
        const w = window.open(url, '_blank', 'noopener,noreferrer')
        if (w && typeof w.blur === 'function') w.blur()
        window.focus()
        setTimeout(() => URL.revokeObjectURL(url), 5000)
      }
      setCalled(prev => ({ ...prev, [routeId]: nextState }))
      localStorage.setItem(calledLsKey(routeId), nextState ? '1' : '0')
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err))
    }
  }

  const markCalledBulk = (routeIds: string[]) => {
    const updates: Record<string, boolean> = {}
    for (const rid of routeIds) {
      updates[rid] = true
      localStorage.setItem(calledLsKey(rid), '1')
    }
    setCalled(prev => ({ ...prev, ...updates }))
    setBulkCalled(true)
  }

  const clearCalledBulk = (routeIds: string[]) => {
    const updates: Record<string, boolean> = {}
    for (const rid of routeIds) {
      updates[rid] = false
      localStorage.removeItem(calledLsKey(rid))
    }
    setCalled(prev => ({ ...prev, ...updates }))
    setBulkCalled(false)
  }

  useEffect(() => {
    setDraftFleet({})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fleetByHorizon.map(r => r.h.join(',')).join('|')])

  // detect unsaved changes in "Готово к отгрузке"
  useEffect(() => {
    const dirty = rows.some(r => {
      const val = draftReady[r.routeId] ?? String(r.readyToShip)
      return String(val) !== String(r.readyToShip)
    })
    onReadyDirtyChange?.(dirty)
  }, [rows, draftReady, onReadyDirtyChange])

  return (
    <div className="space-y-4">
    <div className="bg-surface rounded-lg border border-border overflow-hidden">
      {/* Table header bar */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between gap-3 shrink-0">
        <span className="section-label">
          {warehouse?.name ?? '—'} · Маршруты ({warehouseRoutes.length})
        </span>
        {loading && (
          <span className="flex items-center gap-1.5 text-xs text-muted">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
            Расчёт прогноза…
          </span>
        )}
        {dispatchResult && !loading && (
          <span className="text-xs text-muted">
            Итого:{' '}
            <span className="text-accent font-mono font-semibold">
              {dispatchResult.total_cost.toLocaleString('ru-RU')} ₽
            </span>
          </span>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 text-xs text-status-red bg-status-red/5 border-b border-border">
          {error}
        </div>
      )}

      {/* Table with loading overlay */}
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-surface/70 backdrop-blur-[1px] flex flex-col items-center justify-center z-10 gap-2">
            <Loader2 className="w-7 h-7 animate-spin text-accent" />
            <span className="text-xs text-muted">Загрузка плана развозки…</span>
          </div>
        )}

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Маршрут</TableHead>
              <TableHead className="text-right">Готово к отгрузке</TableHead>
              <TableHead className="text-right">Прогноз на 2ч</TableHead>
              <TableHead className="text-right">Прогноз 2–4ч</TableHead>
              <TableHead className="text-right">Прогноз 4–6ч</TableHead>
              <TableHead className="text-right">Статус</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted py-10 text-sm">
                  Нет маршрутов для данного склада
                </TableCell>
              </TableRow>
            ) : (
              <>
                {rows.map(row => (
                  <TableRow
                    key={row.routeId}
                    selected={row.routeId === selectedRouteId}
                    onClick={() => onSelectRoute(row.routeId)}
                    className="cursor-pointer"
                  >
                    <TableCell>
                      <div>
                        <span className="font-mono text-[11px] text-muted">#{row.routeId}</span>
                        <div className="text-sm text-foreground font-medium">
                          {row.fromCity} → {row.toCity}
                        </div>
                        <div className="text-[11px] text-muted">{row.distanceKm} км</div>
                      </div>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      <input
                        type="number"
                        min="0"
                        value={draftReady[row.routeId] ?? String(row.readyToShip)}
                        onClick={e => e.stopPropagation()}
                        onChange={e => {
                          const value = e.target.value
                          setDraftReady(prev => ({ ...prev, [row.routeId]: value }))
                        }}
                        onBlur={() => {
                          const parsed = Number(draftReady[row.routeId] ?? row.readyToShip)
                          const nextValue = Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : 0
                          setDraftReady(prev => ({ ...prev, [row.routeId]: String(nextValue) }))
                          if (nextValue !== row.readyToShip) {
                            onChangeReadyToShip(row.routeId, nextValue)
                          }
                        }}
                        className="w-24 h-8 rounded bg-elevated border border-border px-2 text-right text-sm text-status-green font-semibold focus:outline-none focus:border-accent"
                      />
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {row.h0 !== null
                        ? <span className={forecastColor(row.h0, row.h0Leftover, row.h0Vehicles)} title={row.h0Lo !== null ? `ДИ: ${fmt(Math.round(row.h0Lo))} — ${fmt(Math.round(row.h0Hi ?? row.h0))}` : undefined}>
                            {fmt(Math.round(row.h0))}
                            {row.h0Hi !== null && row.h0Hi > row.h0
                              ? <span className="text-[9px] text-muted/70 ml-0.5">±{fmt(Math.round(row.h0Hi - row.h0))}</span>
                              : null}
                          </span>
                        : <span className="text-muted">—</span>}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {row.h1 !== null
                        ? <span className={forecastColor(row.h1, row.h1Leftover, row.h1Vehicles)} title={row.h1Lo !== null ? `ДИ: ${fmt(Math.round(row.h1Lo))} — ${fmt(Math.round(row.h1Hi ?? row.h1))}` : undefined}>
                            {fmt(Math.round(row.h1))}
                            {row.h1Hi !== null && row.h1Hi > row.h1
                              ? <span className="text-[9px] text-muted/70 ml-0.5">±{fmt(Math.round(row.h1Hi - row.h1))}</span>
                              : null}
                          </span>
                        : <span className="text-muted">—</span>}
                    </TableCell>
                <TableCell className="text-right font-mono">
                  {row.h2 !== null
                    ? <span className={forecastColor(row.h2, row.h2Leftover, row.h2Vehicles)} title={row.h2Lo !== null ? `ДИ: ${fmt(Math.round(row.h2Lo))} — ${fmt(Math.round(row.h2Hi ?? row.h2))}` : undefined}>
                        {fmt(Math.round(row.h2))}
                        {row.h2Hi !== null && row.h2Hi > row.h2
                          ? <span className="text-[9px] text-muted/70 ml-0.5">±{fmt(Math.round(row.h2Hi - row.h2))}</span>
                          : null}
                      </span>
                    : <span className="text-muted">—</span>}
                </TableCell>
                <TableCell className="text-right">
                  <button
                    className={`px-3 py-1.5 text-xs rounded transition-colors ${
                      called[row.routeId]
                        ? 'bg-muted text-foreground/60 hover:bg-muted/80'
                        : 'bg-accent text-background hover:opacity-90'
                    }`}
                    onClick={async (e) => {
                      e.stopPropagation()
                      await handleCall(row.routeId)
                    }}
                  >
                    {called[row.routeId] ? 'Вызвано' : 'Вызвать'}
                  </button>
                </TableCell>
              </TableRow>
            ))}
            <TableRow className="bg-elevated/60 font-semibold border-t-2 border-border">
              <TableCell>
                <div className="text-sm text-foreground font-semibold">Итого</div>
                  </TableCell>
                  <TableCell className="text-right font-mono text-status-green font-semibold">
                    {fmt(rows.reduce((s, r) => s + r.readyToShip, 0))}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {rows.some(r => r.h0 !== null)
                      ? <span className="text-foreground font-semibold">{fmt(Math.round(rows.reduce((s, r) => s + (r.h0 ?? 0), 0)))}</span>
                      : <span className="text-muted">—</span>}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {rows.some(r => r.h1 !== null)
                      ? <span className="text-foreground font-semibold">{fmt(Math.round(rows.reduce((s, r) => s + (r.h1 ?? 0), 0)))}</span>
                      : <span className="text-muted">—</span>}
                  </TableCell>
              <TableCell className="text-right font-mono">
                {rows.some(r => r.h2 !== null)
                  ? <span className="text-foreground font-semibold">{fmt(Math.round(rows.reduce((s, r) => s + (r.h2 ?? 0), 0)))}</span>
                  : <span className="text-muted">—</span>}
              </TableCell>
              <TableCell className="text-right">
                {onCallAllRoutes && (
                  <button
                    className={`px-3 py-1.5 text-xs rounded transition-colors ${
                      bulkCalled ? 'bg-muted text-foreground/70 hover:bg-muted/80' : 'bg-accent text-background hover:opacity-90'
                    }`}
                    onClick={async (e) => {
                      e.stopPropagation()
                      const ids = rows.map(r => r.routeId)
                      if (bulkCalled) {
                        clearCalledBulk(ids)
                      } else {
                        await onCallAllRoutes(ids)
                        markCalledBulk(ids)
                      }
                    }}
                  >
                    {bulkCalled ? 'Отменить' : 'Вызвать все'}
                  </button>
                )}
              </TableCell>
            </TableRow>
          </>
        )}
      </TableBody>
        </Table>
      </div>
    </div>

      {/* Coverage shortfall warning */}
      {hasShortfall && (
        <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg border border-status-red/50 bg-status-red/10 text-status-red text-xs">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Есть остатки на одном или нескольких горизонтах.</span>
          </div>
        </div>
      )}

      {/* Fleet availability by horizon */}
      {fleetByHorizon.length > 0 && (
        <div className="bg-surface rounded-lg border border-border overflow-hidden">
          <div className="px-4 py-2.5 border-b border-border">
            <span className="section-label">Доступные ТС по горизонтам</span>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Тип ТС</TableHead>
                {HORIZON_LABELS.map(label => (
                  <TableHead key={label} className="text-right">{label}</TableHead>
                ))}
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
                    return (
                      <TableCell key={i} className="text-right">
                        {onFleetChange ? (
                          <input
                            type="number"
                            min="0"
                            value={draftVal}
                            disabled={isSaving}
                            onChange={e => setDraftFleet(prev => ({ ...prev, [k]: e.target.value }))}
                            onFocus={() => setDraftFleet(prev => ({ ...prev, [k]: String(count) }))}
                            onBlur={async () => {
                              const parsed = Number(draftVal)
                              const next = Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : count
                              setDraftFleet(prev => ({ ...prev, [k]: String(next) }))
                              if (next !== count) {
                                setSavingFleet(prev => ({ ...prev, [k]: true }))
                                try {
                                  await onFleetChange(row.type, i as 0|1|2|3, next)
                                } finally {
                                  setSavingFleet(prev => ({ ...prev, [k]: false }))
                                }
                              }
                            }}
                            className={`w-16 h-7 rounded bg-elevated border border-border px-2 text-right text-sm font-mono focus:outline-none focus:border-accent disabled:opacity-50 ${
                              count === 0 ? 'text-status-red' : 'text-foreground'
                            }`}
                          />
                        ) : (
                          <span className={`font-mono text-sm ${count === 0 ? 'text-status-red' : 'text-foreground'}`}>
                            {count}
                          </span>
                        )}
                      </TableCell>
                    )
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

    </div>
  )
}
