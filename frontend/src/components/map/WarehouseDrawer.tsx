import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { X, ArrowRight, Loader2 } from 'lucide-react'
import type { Warehouse, RouteDistance, ForecastPoint, ApiDispatchResponse, SankeyData } from '../../types'
import { Badge } from '../ui/badge'
import type { BadgeVariant } from '../ui/badge'
import { ForecastChart } from './ForecastChart'
import { SankeyChart } from './SankeyChart'
import { fmt, horizonDisplayLabel, getFutureHorizonKeys } from '../../lib/utils'
import { Button } from '../ui/button'
import { useSimulationContext } from '../../context/SimulationContext'
import { postDispatch, getWarehouseSankey } from '../../api'
import { MetricDetailModal } from '../optimizer/MetricDetailModal'

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
  const { vehicleTypes, analysisDateTime, incomingVehicles, setWarehouseStatus, riskSettings, warehouseStatuses, setWarehouseMetrics } = useSimulationContext()
  const granularity = riskSettings.granularity

  // ── Forecast fetch on warehouse open ────────────────────────────────────
  // Dispatch fetch on warehouse open (LS-backed) ────────────────────────
  const [dispatchResult, setDispatchResult] = useState<ApiDispatchResponse | null>(null)
  const [dispatchLoading, setDispatchLoading] = useState(false)
  const [metricDetail, setMetricDetail] = useState<'p_cover' | 'fill_rate' | 'cpo' | 'fleet_utilization' | 'capacity_shortfall' | null>(null)

  function deriveWarehouseStatus(result: ApiDispatchResponse): 'ok' | 'warning' | 'critical' {
    const shortfall = result.metrics?.fleet_capacity_shortfall ?? 0
    let hasFullyMissed = false, hasPartiallyMissed = false
    for (const rp of result.routes) {
      for (const row of rp.plan) {
        if (row.demand_new > 0 && row.vehicles_count === 0) hasFullyMissed = true
        else if (row.leftover_stock >= 1) hasPartiallyMissed = true
      }
    }
    if (shortfall > 0 && hasFullyMissed) return 'critical'
    if (shortfall > 0 && hasPartiallyMissed) return 'warning'
    return 'ok'
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
      if (cached.metrics) setWarehouseMetrics(warehouse.id, cached.metrics)
    } else {
      postDispatch({
        warehouse_id: warehouse.id,
        timestamp: ts,
        incoming_vehicles: incomingVehicles.length > 0 ? incomingVehicles : undefined,
        confidence_level: riskSettings.confidenceLevel,
        granularity: granularity,
      })
        .then(result => {
          lsSet(key, result)
          setDispatchResult(result)
          setWarehouseStatus(warehouse.id, deriveWarehouseStatus(result))
          if (result.metrics) setWarehouseMetrics(warehouse.id, result.metrics)
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

  // Sankey: fetch real status data from backend
  const [sankeyData, setSankeyData] = useState<SankeyData>({ nodes: [], links: [] })
  useEffect(() => {
    if (!warehouse) { setSankeyData({ nodes: [], links: [] }); return }
    const rid = selectedRouteId && selectedRoute ? selectedRouteId : undefined
    getWarehouseSankey(warehouse.id, analysisDateTime.replace('T', ' ') + ':00', rid)
      .then(setSankeyData)
      .catch(() => setSankeyData({ nodes: [], links: [] }))
  }, [warehouse?.id, analysisDateTime, selectedRouteId, selectedRoute])

  // Build chart data from dispatch results (same numbers as table above)
  const futureHorizonKeys = useMemo(() => getFutureHorizonKeys(dispatchResult, granularity), [dispatchResult, granularity])

  const chartForecastData = useMemo<ForecastPoint[]>(() => {
    if (!dispatchResult) return []

    const horizonKeys = [
      { key: 'ready', label: 'Сейчас' },
      ...futureHorizonKeys.map(k => ({ key: k, label: horizonDisplayLabel(k) })),
    ]

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
  }, [dispatchResult, selectedRouteId, futureHorizonKeys, granularity])

  if (!warehouse) return null

  // Build per-route forecast rows from dispatch result
  const routeForecastRows = warehouseRoutes.map(r => {
    const rp = dispatchResult?.routes.find(x => x.route_id === r.id)
    const horizons = futureHorizonKeys.map(hk => {
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
              <Badge variant={STATUS_BADGE[(warehouseStatuses[warehouse.id] ?? warehouse.status) as Warehouse['status']] ?? STATUS_BADGE[warehouse.status]}>
                {STATUS_LABEL[(warehouseStatuses[warehouse.id] ?? warehouse.status) as Warehouse['status']] ?? STATUS_LABEL[warehouse.status]}
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
                    {futureHorizonKeys.map(hk => (
                      <th key={hk} className="px-3 py-2 text-right section-label">{horizonDisplayLabel(hk)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {routeForecastRows.length === 0 ? (
                    <tr><td colSpan={2 + futureHorizonKeys.length} className="px-3 py-4 text-center text-muted text-xs">Нет маршрутов</td></tr>
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
                      {futureHorizonKeys.map((hk, i) => {
                        const total = routeForecastRows.reduce((s, { horizons }) => {
                          const h = horizons[i]
                          return s + (h?.demand ?? 0)
                        }, 0)
                        return (
                          <td key={hk} className="px-3 py-2 text-right font-mono">
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

          {/* ── Бизнес-метрики ───────────────────────────────────────────── */}
          {(dispatchResult?.metrics || dispatchLoading) && (() => {
            if (dispatchLoading && !dispatchResult?.metrics) {
              return (
                <section>
                  <div className="section-label mb-2">Бизнес-метрики</div>
                  <div className="bg-elevated rounded-lg px-4 py-3">
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 animate-pulse">
                      {[...Array(5)].map((_, i) => (
                        <div key={i} className="flex flex-col gap-1.5">
                          <div className="h-3 bg-surface rounded w-3/4" />
                          <div className="h-6 bg-surface rounded w-1/2" />
                          <div className="h-2.5 bg-surface rounded w-2/3" />
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )
            }
            const m = dispatchResult!.metrics!
            const horizonLabels = m.horizon_labels ?? ['A: сейчас', 'B: +2ч', 'C: +4ч', 'D: +6ч']
            const pColor = m.p_cover >= 0.9 ? 'text-status-green' : m.p_cover >= 0.7 ? 'text-status-yellow' : 'text-status-red'
            const utilRatio = m.fleet_utilization_ratio
            const utilColor = utilRatio == null ? 'text-[#8A9CC0]'
              : utilRatio <= 0.8 ? 'text-status-green'
              : utilRatio <= 1.0 ? 'text-status-yellow'
              : 'text-status-red'
            const shortfall = m.fleet_capacity_shortfall
            const shortColor = shortfall == null ? 'text-[#8A9CC0]'
              : shortfall <= 0 ? 'text-status-green'
              : shortfall <= 50 ? 'text-status-yellow'
              : 'text-status-red'
            return (
              <section>
                <div className="section-label mb-2">Бизнес-метрики <span className="text-[10px] text-[#8A9CC0] font-normal ml-1">(клик → детали)</span></div>
                <div className={`rounded-lg bg-[#1A2040] border border-[#2A3560] px-4 py-3 transition-opacity ${dispatchLoading ? 'opacity-50' : ''}`}>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                    <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('p_cover')}>
                      <span className="text-[11px] text-[#8A9CC0] leading-tight">Вероятность покрытия</span>
                      <span className={`text-lg font-bold font-mono ${pColor}`}>{(m.p_cover * 100).toFixed(1)}%</span>
                      <div className="flex flex-wrap gap-x-2 gap-y-0">
                        {m.p_cover_by_horizon.map((p, i) => i === 0 ? null : (
                          <span key={i} className="text-[10px] text-[#8A9CC0]">
                            {horizonDisplayLabel(horizonLabels[i])}&nbsp;<span className={p >= 0.9 ? 'text-status-green' : p >= 0.7 ? 'text-status-yellow' : 'text-status-red'}>{(p*100).toFixed(0)}%</span>
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('fill_rate')}>
                      <span className="text-[11px] text-[#8A9CC0] leading-tight">Загрузка ТС</span>
                      <span className={`text-lg font-bold font-mono ${m.fill_rate >= 0.8 ? 'text-status-green' : m.fill_rate >= 0.5 ? 'text-status-yellow' : 'text-status-red'}`}>
                        {(m.fill_rate * 100).toFixed(1)}%
                      </span>
                      <span className="text-[10px] text-[#8A9CC0]">Отправлено / вместимость</span>
                    </div>
                    <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('cpo')}>
                      <span className="text-[11px] text-[#8A9CC0] leading-tight">Стоимость доставки</span>
                      <span className="text-lg font-bold font-mono text-white">{m.cpo.toFixed(0)} ₽</span>
                      <span className="text-[10px] text-[#8A9CC0]">За единицу отправлено</span>
                    </div>
                    <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('fleet_utilization')}>
                      <span className="text-[11px] text-[#8A9CC0] leading-tight">Утилизация флота</span>
                      <span className={`text-lg font-bold font-mono ${utilColor}`}>
                        {utilRatio != null ? utilRatio.toFixed(2) : '—'}
                      </span>
                      <span className="text-[10px] text-[#8A9CC0]">
                        {m.required_capacity_units != null && m.available_capacity_units != null
                          ? `${Math.round(m.required_capacity_units)} / ${Math.round(m.available_capacity_units)} ед.`
                          : 'Треб. / доступно'}
                      </span>
                    </div>
                    <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('capacity_shortfall')}>
                      <span className="text-[11px] text-[#8A9CC0] leading-tight">Нехватка вместимости</span>
                      <span className={`text-lg font-bold font-mono ${shortColor}`}>
                        {shortfall != null
                          ? (shortfall > 0 ? `+${Math.round(shortfall)}` : Math.round(shortfall).toString())
                          : '—'} ед.
                      </span>
                      <span className="text-[10px] text-[#8A9CC0]">Треб. − доступно</span>
                    </div>
                  </div>
                </div>
                <MetricDetailModal metricKey={metricDetail} metrics={m} onClose={() => setMetricDetail(null)} />
              </section>
            )
          })()}

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
