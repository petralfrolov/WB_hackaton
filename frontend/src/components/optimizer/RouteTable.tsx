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
  vehicleTypes?: VehicleType[]
  incomingVehicles?: ApiIncomingVehicle[]
  onFleetChange?: (vehicleType: string, horizonIdx: 0 | 1 | 2 | 3, newCount: number) => Promise<void>
}

function forecastColor(forecast: number, readyToShip: number): string {
  if (forecast <= 0) return 'text-muted'
  if (forecast > readyToShip) return 'text-status-red font-semibold'
  return 'text-status-yellow font-semibold'
}

interface RouteRow {
  routeId: string
  fromCity: string
  toCity: string
  distanceKm: number
  readyToShip: number
  h0: number | null  // 0–2h
  h1: number | null  // 2–4h
  h2: number | null  // 4–6h
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
  vehicleTypes = [],
  incomingVehicles = [],
  onFleetChange,
}: RouteTableProps) {
  const [draftReady, setDraftReady] = useState<Record<string, string>>({})
  const [draftFleet, setDraftFleet] = useState<Record<string, string>>({})
  const [savingFleet, setSavingFleet] = useState<Record<string, boolean>>({})

  useEffect(() => {
    const nextDrafts = Object.fromEntries(warehouseRoutes.map(route => [route.id, String(route.readyToShip)]))
    setDraftReady(nextDrafts)
  }, [warehouseRoutes])

  // Build lookup: route_id → forecast by horizon
  const forecastMap = new Map<string, { h0: number; h1: number; h2: number }>()
  if (dispatchResult) {
    for (const rp of dispatchResult.routes) {
      const hB = rp.plan.find(r => r.horizon === 'B: +2h')
      const hC = rp.plan.find(r => r.horizon === 'C: +4h')
      const hD = rp.plan.find(r => r.horizon === 'D: +6h')
      forecastMap.set(rp.route_id, {
        h0: hB?.demand_new ?? 0,
        h1: hC?.demand_new ?? 0,
        h2: hD?.demand_new ?? 0,
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
    }
  })

  const hasShortfall = dispatchResult !== null && dispatchResult.routes.some(r => r.coverage_min < -0.5)

  const fleetByHorizon = computeFleetByHorizon(vehicleTypes, incomingVehicles)

  useEffect(() => {
    setDraftFleet({})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fleetByHorizon.map(r => r.h.join(',')).join('|')])

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
              rows.map(row => (
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
                      ? <span className={forecastColor(row.h0, row.readyToShip)}>{fmt(Math.round(row.h0))}</span>
                      : <span className="text-muted">—</span>}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {row.h1 !== null
                      ? <span className={forecastColor(row.h1, row.readyToShip)}>{fmt(Math.round(row.h1))}</span>
                      : <span className="text-muted">—</span>}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {row.h2 !== null
                      ? <span className={forecastColor(row.h2, row.readyToShip)}>{fmt(Math.round(row.h2))}</span>
                      : <span className="text-muted">—</span>}
                  </TableCell>
                </TableRow>
              ))
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
            <span className="font-semibold">Недостаточно транспорта для полного покрытия спроса.</span>
            {' '}Добавьте дополнительные ТС или скорректируйте парк во вкладке «Бизнес-правила».
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
