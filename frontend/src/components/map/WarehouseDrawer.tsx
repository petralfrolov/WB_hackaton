import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { X, ArrowRight, Loader2 } from 'lucide-react'
import type { Warehouse, RouteDistance, ForecastPoint, ApiDispatchResponse } from '../../types'
import { Badge } from '../ui/badge'
import type { BadgeVariant } from '../ui/badge'
import { ForecastChart } from './ForecastChart'
import { SankeyChart } from './SankeyChart'
import { fmt, makeSankey } from '../../lib/utils'
import { Button } from '../ui/button'
import { useSimulationContext } from '../../context/SimulationContext'
import { postDispatch } from '../../api'

// ── LS cache helpers (same key scheme as OptimizerPage) ──────────────────────
const LS_PREFIX = 'dispatch_cache__'
function lsGet(key: string): ApiDispatchResponse | null {
  try { const r = localStorage.getItem(LS_PREFIX + key); return r ? JSON.parse(r) : null } catch { return null }
}
function lsSet(key: string, v: ApiDispatchResponse) {
  try { localStorage.setItem(LS_PREFIX + key, JSON.stringify(v)) } catch { /* quota */ }
}

interface WarehouseDrawerProps {
  warehouse: Warehouse | null
  onClose: () => void
  routes: RouteDistance[]
}

const STATUS_LABEL: Record<Warehouse['status'], string> = {
  ok: 'В норме',
  warning: 'Предупреждение',
  critical: 'Критично',
}

const STATUS_BADGE: Record<Warehouse['status'], BadgeVariant> = {
  ok: 'ok',
  warning: 'warning',
  critical: 'critical',
}

export function WarehouseDrawer({ warehouse, onClose, routes }: WarehouseDrawerProps) {
  const drawerRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  // '' = aggregate (все маршруты), otherwise specific route id
  const [selectedRouteId, setSelectedRouteId] = useState('')
  const { vehicleTypes, analysisDateTime, incomingVehicles, setWarehouseStatus, riskSettings } = useSimulationContext()

  // ── Forecast fetch on warehouse open ────────────────────────────────────
  // Dispatch fetch on warehouse open (LS-backed) ────────────────────────
  const [dispatchResult, setDispatchResult] = useState<ApiDispatchResponse | null>(null)
  const [dispatchLoading, setDispatchLoading] = useState(false)

  function deriveWarehouseStatus(result: ApiDispatchResponse): 'ok' | 'warning' | 'critical' {
    let hasCritical = false, hasWarning = false
    for (const rp of result.routes) {
      for (const row of rp.plan) {
        if (row.demand_new > 0 && row.vehicles_count === 0) hasCritical = true
        else if (row.leftover_stock >= 1) hasWarning = true
      }
    }
    return hasCritical ? 'critical' : hasWarning ? 'warning' : 'ok'
  }

  useEffect(() => {
    if (!warehouse) return
    setDispatchResult(null)
    setDispatchLoading(true)
    const ts = analysisDateTime.replace('T', ' ') + ':00'
    const key = `${warehouse.id}__${analysisDateTime}`

    // Dispatch plan (LS cache → backend)
    const cached = lsGet(key)
    if (cached) {
      setDispatchResult(cached)
      setDispatchLoading(false)
      setWarehouseStatus(warehouse.id, deriveWarehouseStatus(cached))
    } else {
      postDispatch({
        warehouse_id: warehouse.id,
        timestamp: ts,
        incoming_vehicles: incomingVehicles.length > 0 ? incomingVehicles : undefined,
        confidence_level: riskSettings.confidenceLevel,
      })
        .then(result => {
          lsSet(key, result)
          setDispatchResult(result)
          setWarehouseStatus(warehouse.id, deriveWarehouseStatus(result))
        })
        .catch(() => {})
        .finally(() => setDispatchLoading(false))
    }
  }, [warehouse?.id, analysisDateTime]) // eslint-disable-line react-hooks/exhaustive-deps

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const warehouseRoutes = useMemo(
    () => (warehouse ? routes.filter(r => r.fromId === warehouse.id) : []),
    [routes, warehouse],
  )

  // Reset to "all" when warehouse changes
  useEffect(() => {
    setSelectedRouteId('')
  }, [warehouse?.id])

  const selectedRoute = warehouseRoutes.find(r => r.id === selectedRouteId) ?? null

  // Sankey: aggregate for warehouse or per-route
  const sankeyData = useMemo(() => {
    if (!warehouse) return makeSankey(0)
    if (!selectedRouteId || !selectedRoute) {
      // Aggregate: use the warehouse-level sankey (sum across routes)
      return warehouse.sankeyData
    }
    // Per-route: scale down proportionally from warehouse total
    const perRouteValue = Math.round(warehouse.readyToShip * 4 / Math.max(warehouseRoutes.length, 1))
    return makeSankey(perRouteValue)
  }, [selectedRouteId, selectedRoute, warehouse, warehouseRoutes.length])

  // Build chart data from dispatch results (same numbers as table above)
  const chartForecastData = useMemo<ForecastPoint[]>(() => {
    const horizonKeys = [
      { key: 'ready', label: 'Сейчас' },
      { key: 'B: +2h', label: '+2ч' },
      { key: 'C: +4h', label: '+4ч' },
      { key: 'D: +6h', label: '+6ч' },
    ]
    if (dispatchResult) {
      return horizonKeys.map(({ key, label }) => {
        let value = 0
        let lower = 0
        let upper = 0
        if (key === 'ready') {
          if (selectedRouteId) {
            value = warehouseRoutes.find(r => r.id === selectedRouteId)?.readyToShip ?? 0
          } else {
            value = warehouseRoutes.reduce((s, r) => s + r.readyToShip, 0)
          }
          lower = value
          upper = value
        } else {
          if (selectedRouteId) {
            const rp = dispatchResult.routes.find(r => r.route_id === selectedRouteId)
            const row = rp?.plan.find(p => p.horizon === key)
            value = Math.round(row?.demand_new ?? 0)
            lower = Math.round(row?.demand_lower ?? value)
            upper = Math.round(row?.demand_upper ?? value)
          } else {
            for (const rp of dispatchResult.routes) {
              const row = rp.plan.find(p => p.horizon === key)
              value += Math.round(row?.demand_new ?? 0)
              lower += Math.round(row?.demand_lower ?? 0)
              upper += Math.round(row?.demand_upper ?? 0)
            }
          }
        }
        return { time: label, value, lower, upper }
      })
    }
    // No dispatch data yet — return empty (chart shows loading state)
    return []
  }, [dispatchResult, selectedRouteId])

  if (!warehouse) return null

  // Build per-route forecast rows from dispatch result
  const routeForecastRows = warehouseRoutes.map(r => {
    const rp = dispatchResult?.routes.find(x => x.route_id === r.id)
    const horizons = ['B: +2h', 'C: +4h', 'D: +6h'].map(hk => {
      const row = rp?.plan.find(p => p.horizon === hk)
      return row
        ? { demand: row.demand_new, vehicles: row.vehicles_count, leftover: row.leftover_stock }
        : null
    })
    return { route: r, horizons }
  })

  function fcColor(demand: number, vehicles: number | null, leftover: number | null): string {
    if (demand <= 0) return 'text-muted'
    if (vehicles === null) return 'text-foreground'
    if (vehicles === 0) return 'text-status-red font-semibold'
    if ((leftover ?? 0) >= 1) return 'text-status-yellow font-semibold'
    return 'text-foreground'
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30 z-10"
        onClick={onClose}
      />

      {/* Drawer panel */}
      <div
        ref={drawerRef}
        className="absolute right-0 top-0 h-full w-[680px] bg-surface border-l border-border z-20 flex flex-col overflow-y-auto drawer-enter"
      >
        {/* Header */}
        <div className="flex items-start justify-between px-6 py-4 border-b border-border shrink-0">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-bold text-foreground">{warehouse.name}</h2>
              <Badge variant={STATUS_BADGE[warehouse.status]}>
                {STATUS_LABEL[warehouse.status]}
              </Badge>
            </div>
            <p className="text-muted text-sm mt-0.5">{warehouse.city}</p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-elevated text-muted hover:text-foreground transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="px-6 py-5 space-y-6 flex-1">
          {/* ── Routes Table ─────────────────────────────────────────────────── */}
          <section>
            <div className="section-label mb-2 flex items-center gap-2">
              Маршруты и прогноз отгрузок
              {dispatchLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />}
            </div>
            <div className="bg-elevated rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-3 py-2 text-left section-label">Маршрут</th>
                    <th className="px-3 py-2 text-right section-label">К отгрузке</th>
                    <th className="px-3 py-2 text-right section-label">+2ч</th>
                    <th className="px-3 py-2 text-right section-label">+4ч</th>
                    <th className="px-3 py-2 text-right section-label">+6ч</th>
                  </tr>
                </thead>
                <tbody>
                  {routeForecastRows.length === 0 ? (
                    <tr><td colSpan={5} className="px-3 py-4 text-center text-muted text-xs">Нет маршрутов</td></tr>
                  ) : routeForecastRows.map(({ route, horizons }) => (
                    <tr
                      key={route.id}
                      className={`border-b border-border/50 last:border-0 cursor-pointer transition-colors ${selectedRouteId === route.id ? 'bg-accent/10' : 'hover:bg-surface'}`}
                      onClick={() => setSelectedRouteId(prev => prev === route.id ? '' : route.id)}
                    >
                      <td className="px-3 py-2">
                        <div className="text-xs text-muted font-mono">#{route.id}</div>
                        <div className="text-foreground">{route.fromCity} → {route.toCity}</div>
                        <div className="text-[11px] text-muted">{route.distanceKm} км</div>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-accent font-semibold">{fmt(route.readyToShip)}</td>
                      {horizons.map((h, i) => (
                        <td key={i} className="px-3 py-2 text-right font-mono">
                          {h
                            ? <span className={fcColor(h.demand, h.vehicles, h.leftover)}>{fmt(Math.round(h.demand))}</span>
                            : <span className="text-muted">—</span>}
                        </td>
                      ))}
                    </tr>
                  ))}                  {routeForecastRows.length > 0 && (
                    <tr className="border-t-2 border-border bg-elevated/50">
                      <td className="px-3 py-2 text-foreground font-semibold">Итого</td>
                      <td className="px-3 py-2 text-right font-mono text-accent font-semibold">
                        {fmt(warehouseRoutes.reduce((s, r) => s + r.readyToShip, 0))}
                      </td>
                      {[0, 1, 2].map(i => {
                        const total = routeForecastRows.reduce((s, { horizons }) => {
                          const h = horizons[i]
                          return s + (h?.demand ?? 0)
                        }, 0)
                        return (
                          <td key={i} className="px-3 py-2 text-right font-mono">
                            {dispatchResult
                              ? <span className="text-foreground font-semibold">{fmt(Math.round(total))}</span>
                              : <span className="text-muted">—</span>}
                          </td>
                        )
                      })}
                    </tr>
                  )}                </tbody>
              </table>
            </div>
            {warehouseRoutes.length > 0 && (
              <div className="text-[11px] text-muted mt-1">Кликните на маршрут, чтобы отфильтровать диаграмму ниже</div>
            )}
          </section>

          {/* ── Детализация button ───────────────────────────────────────────── */}
          <div className="flex justify-end">
            <Button
              size="sm"
              variant="outline"
              onClick={() => { onClose(); navigate(`/optimizer?warehouseId=${warehouse.id}`) }}
            >
              <ArrowRight className="w-3.5 h-3.5 mr-1" />
              Детализация в Оптимизаторе
            </Button>
          </div>

          {/* ── Forecast Chart ───────────────────────────────────────────────── */}
          <section>
            {/* Режим отображения filter – moved here, above chart */}
            <div className="bg-elevated rounded-lg p-3 mb-3 border border-border/60">
              <label className="text-[11px] text-muted block mb-1">Режим отображения</label>
              <select
                value={selectedRouteId}
                onChange={e => setSelectedRouteId(e.target.value)}
                className="w-full h-8 rounded bg-surface border border-border px-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
              >
                {warehouseRoutes.length === 0
                  ? <option value="">Нет маршрутов из этого склада</option>
                  : <>
                      <option value="">Все маршруты (сумма)</option>
                      {warehouseRoutes.map(route => (
                        <option key={route.id} value={route.id}>
                          #{route.id}: {route.fromCity} → {route.toCity} · {route.distanceKm} км
                        </option>
                      ))}
                    </>
                }
              </select>
            </div>
            <div className="section-label mb-2 flex items-center gap-2">
              Прогноз отгрузок — ближайшие 6 часов
              {dispatchLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />}
              {selectedRoute && <span className="text-[11px] text-muted font-normal">· {selectedRoute.fromCity} → {selectedRoute.toCity}</span>}
            </div>
            <div className="bg-elevated rounded-lg p-3">
              {chartForecastData.length > 0
                ? <ForecastChart data={chartForecastData} scale={1} />
                : dispatchLoading
                  ? <div className="h-[200px] flex items-center justify-center text-muted text-xs">Загрузка прогноза…</div>
                  : <div className="h-[200px] flex items-center justify-center text-muted text-xs">Нет данных прогноза</div>
              }
            </div>
          </section>

          {/* ── Sankey ──────────────────────────────────────────────────────── */}
          <section>
            <div className="section-label mb-3">Движение товаров по статусам</div>
            <div className="bg-elevated rounded-lg p-4 overflow-x-auto">
              <SankeyChart data={sankeyData} width={588} height={260} />
            </div>
          </section>

          {/* ── Vehicle Park ─────────────────────────────────────────────────── */}
          <section>
            <div className="section-label mb-3">Парк транспортных средств</div>
            <div className="bg-elevated rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-3 py-2 text-left section-label">ТС</th>
                    <th className="px-3 py-2 text-right section-label">Ёмкость</th>
                    <th className="px-3 py-2 text-right section-label">₽/км</th>
                    <th className="px-3 py-2 text-right section-label">Доступно</th>
                  </tr>
                </thead>
                <tbody>
                  {vehicleTypes.map(v => (
                    <tr key={v.id} className="border-b border-border/50 last:border-0">
                      <td className="px-3 py-2 text-foreground">{v.name}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted">{fmt(v.capacity)}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted">{fmt(v.costPerKm)}</td>
                      <td className="px-3 py-2 text-right font-mono text-accent font-semibold">{v.available}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </>
  )
}
