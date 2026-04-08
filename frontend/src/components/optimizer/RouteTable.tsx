import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react'
import { useEffect, useState, useMemo } from 'react'
import type { Warehouse, RouteDistance, ApiDispatchResponse, VehicleType, ApiIncomingVehicle, Granularity } from '../../types'
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '../ui/table'
import { fmt, makeHorizonLabels, horizonDisplayLabel } from '../../lib/utils'

function computeFleetByHorizon(
  vehicleTypes: VehicleType[],
  incomingVehicles: ApiIncomingVehicle[],
  horizonCount: number,
): Array<{ type: string; h: number[] }> {
  return vehicleTypes.map(vt => {
    const base = vt.available
    const additions = Array(horizonCount).fill(0) as number[]
    for (const iv of incomingVehicles) {
      if (iv.vehicle_type === vt.id && iv.horizon_idx < horizonCount) {
        for (let h = iv.horizon_idx; h < horizonCount; h++) {
          additions[h] += iv.count
        }
      }
    }
    return {
      type: vt.id,
      h: additions.map(a => base + a),
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
  onFleetChange?: (vehicleType: string, horizonIdx: number, newCount: number) => Promise<void>
  onSyncVehicle?: (vehicleType: string) => Promise<void>
  analysisDateTime?: string
  granularity?: Granularity
}

function forecastColor(demand: number, leftover: number | null, vehiclesCount: number | null): string {
  if (demand <= 0) return 'text-muted'
  if (leftover === null) return 'text-foreground' // no dispatch result yet
  if (vehiclesCount === 0) return 'text-status-red font-semibold'   // nothing dispatched
  if (leftover >= 1) return 'text-status-yellow font-semibold'       // partial coverage
  return 'text-foreground'                                           // fully covered
}

interface HorizonCell {
  demand: number
  lo: number
  hi: number
  leftover: number
  vehicles: number
}

interface RouteRow {
  routeId: string
  fromCity: string
  toCity: string
  distanceKm: number
  readyToShip: number
  horizons: (HorizonCell | null)[]  // one per future horizon
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
  onSyncVehicle,
  analysisDateTime = '',
  granularity = 2,
}: RouteTableProps) {
  const calledLsKey = (routeId: string) => `called_${routeId}__${analysisDateTime}`
  const [draftReady, setDraftReady] = useState<Record<string, string>>({})
  const [draftFleet, setDraftFleet] = useState<Record<string, string>>({})
  const [savingFleet, setSavingFleet] = useState<Record<string, boolean>>({})
  const [syncingType, setSyncingType] = useState<Record<string, boolean>>({})
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

  // Derive horizon labels from dispatch response or granularity
  const allHorizonLabels = useMemo(
    () => dispatchResult?.horizon_labels ?? makeHorizonLabels(granularity as Granularity),
    [dispatchResult, granularity],
  )
  // Fleet table always uses 30-min granularity
  const fleetHorizonLabels = useMemo(() => makeHorizonLabels(0.5), [])
  // Future-only labels (skip the first "A: сейчас" / "A: now" label)
  const futureLabels = useMemo(() => allHorizonLabels.slice(1), [allHorizonLabels])

  // Build lookup: route_id → horizon cells
  const forecastMap = useMemo(() => {
    const map = new Map<string, (HorizonCell | null)[]>()
    if (!dispatchResult) return map
    for (const rp of dispatchResult.routes) {
      const cells: (HorizonCell | null)[] = futureLabels.map(label => {
        const row = rp.plan.find(r => r.horizon === label)
        if (!row) return null
        return {
          demand: row.demand_new,
          lo: row.demand_lower,
          hi: row.demand_upper,
          leftover: row.leftover_stock,
          vehicles: row.vehicles_count,
        }
      })
      map.set(rp.route_id, cells)
    }
    return map
  }, [dispatchResult, futureLabels])

  const rows: RouteRow[] = warehouseRoutes.map(r => ({
    routeId: r.id,
    fromCity: r.fromCity,
    toCity: r.toCity,
    distanceKm: r.distanceKm,
    readyToShip: r.readyToShip,
    horizons: forecastMap.get(r.id) ?? futureLabels.map(() => null),
  }))

  const hasShortfall = dispatchResult !== null && dispatchResult.routes.some(r => r.coverage_min < -0.5)

  const fleetByHorizon = computeFleetByHorizon(vehicleTypes, incomingVehicles, fleetHorizonLabels.length)

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
              {futureLabels.map(label => (
                <TableHead key={label} className="text-right">{horizonDisplayLabel(label)}</TableHead>
              ))}
              <TableHead className="text-right">Итого</TableHead>
              <TableHead className="text-right">Статус</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={3 + futureLabels.length} className="text-center text-muted py-10 text-sm">
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
                    {row.horizons.map((cell, i) => (
                      <TableCell key={i} className="text-right font-mono">
                        {cell !== null
                          ? <span className={forecastColor(cell.demand, cell.leftover, cell.vehicles)} title={`ДИ: ${fmt(Math.round(cell.lo))} — ${fmt(Math.round(cell.hi))}`}>
                              {fmt(Math.round(cell.demand))}
                              {cell.hi > cell.demand
                                ? <span className="text-[9px] text-muted/70 ml-0.5">±{fmt(Math.round(cell.hi - cell.demand))}</span>
                                : null}
                            </span>
                          : <span className="text-muted">—</span>}
                      </TableCell>
                    ))}
                    <TableCell className="text-right font-mono font-semibold text-foreground">
                      {row.horizons.some(c => c !== null)
                        ? (() => {
                            const totalDemand = row.readyToShip + row.horizons.reduce((s, c) => s + (c?.demand ?? 0), 0)
                            const ciHi = row.horizons.reduce((s, c) => s + ((c?.hi ?? c?.demand ?? 0) - (c?.demand ?? 0)), 0)
                            return <>
                              {fmt(Math.round(totalDemand))}
                              {ciHi > 0.5
                                ? <span className="text-[9px] text-muted/70 ml-0.5">±{fmt(Math.round(ciHi))}</span>
                                : null}
                            </>
                          })()
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
                  {futureLabels.map((_, i) => (
                    <TableCell key={i} className="text-right font-mono">
                      {rows.some(r => r.horizons[i] !== null)
                        ? (() => {
                            const sumDemand = rows.reduce((s, r) => s + (r.horizons[i]?.demand ?? 0), 0)
                            const sumCi = rows.reduce((s, r) => {
                              const c = r.horizons[i]
                              return s + ((c?.hi ?? c?.demand ?? 0) - (c?.demand ?? 0))
                            }, 0)
                            return <span className="text-foreground font-semibold">
                              {fmt(Math.round(sumDemand))}
                              {sumCi > 0.5
                                ? <span className="text-[9px] text-muted/70 ml-0.5">±{fmt(Math.round(sumCi))}</span>
                                : null}
                            </span>
                          })()
                        : <span className="text-muted">—</span>}
                    </TableCell>
                  ))}
                  <TableCell className="text-right font-mono font-bold text-accent">
                    {rows.some(r => r.horizons.some(c => c !== null))
                      ? (() => {
                          const totalReady = rows.reduce((s, r) => s + r.readyToShip, 0)
                          const totalDemand = totalReady + rows.reduce((s, r) => s + r.horizons.reduce((hs, c) => hs + (c?.demand ?? 0), 0), 0)
                          const totalCi = rows.reduce((s, r) => s + r.horizons.reduce((hs, c) => hs + ((c?.hi ?? c?.demand ?? 0) - (c?.demand ?? 0)), 0), 0)
                          return <>
                            {fmt(Math.round(totalDemand))}
                            {totalCi > 0.5
                              ? <span className="text-[9px] text-muted/70 ml-0.5">±{fmt(Math.round(totalCi))}</span>
                              : null}
                          </>
                        })()
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
                {fleetHorizonLabels.map(label => (
                  <TableHead key={label} className="text-right">{horizonDisplayLabel(label)}</TableHead>
                ))}
                {onSyncVehicle && <TableHead className="text-center">Уст. на все склады</TableHead>}
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
                                  await onFleetChange(row.type, i, next)
                                } finally {
                                  setSavingFleet(prev => ({ ...prev, [k]: false }))
                                }
                              }
                            }}
                            className={`w-16 h-7 rounded bg-elevated border border-border px-2 text-right text-sm font-mono focus:outline-none focus:border-accent disabled:opacity-50 ${
                              count === 0 ? 'text-status-red' : isIncrease ? 'text-status-green' : 'text-foreground'
                            }`}
                          />
                        ) : (
                          <span className={`font-mono text-sm ${count === 0 ? 'text-status-red' : isIncrease ? 'text-status-green' : 'text-foreground'}`}>
                            {count}
                          </span>
                        )}
                      </TableCell>
                    )
                  })}
                  {onSyncVehicle && (
                    <TableCell className="text-center">
                      <button
                        disabled={syncingType[row.type]}
                        onClick={async () => {
                          setSyncingType(prev => ({ ...prev, [row.type]: true }))
                          try {
                            await onSyncVehicle(row.type)
                          } finally {
                            setSyncingType(prev => ({ ...prev, [row.type]: false }))
                          }
                        }}
                        className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded border border-border text-muted hover:text-accent hover:border-accent transition-colors disabled:opacity-50"
                        title="Скопировать количество ТС этого типа на все склады"
                      >
                        <RefreshCw className={`w-3 h-3 ${syncingType[row.type] ? 'animate-spin' : ''}`} />
                      </button>
                    </TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

    </div>
  )
}
