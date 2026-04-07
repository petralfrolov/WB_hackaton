import { useState, useCallback, useEffect, useRef, type MouseEvent as ReactMouseEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { ApiDispatchResponse } from '../types'
import { cn, routeDistanceToApi, makeHorizonLabels, horizonDisplayLabel } from '../lib/utils'
import { RouteTable } from '../components/optimizer/RouteTable'
import { CostBenefitCard } from '../components/optimizer/CostBenefitCard'
import { MetricDetailModal } from '../components/optimizer/MetricDetailModal'
import { useSimulationContext } from '../context/SimulationContext'
import { postDispatch, putRouteDistances, updateVehicle, getVehicles, getIncomingVehicles, putIncomingVehicles, callRoute, syncVehicleAcrossWarehouses } from '../api'
import { apiVehicleToVehicleType } from '../lib/utils'
import { RefreshCw } from 'lucide-react'

export function OptimizerPage() {
  const [searchParams] = useSearchParams()
  const { warehouses, routes, setRoutes, riskSettings, analysisDateTime, setSelectedWarehouseId, incomingVehicles, vehicleTypes, setVehicleTypes, setIncomingVehicles, warehouseStatuses, setWarehouseStatus, refreshAllWarehouses, refreshingAll } = useSimulationContext()
  const refreshingAllRef = useRef(refreshingAll)
  refreshingAllRef.current = refreshingAll

  const [warehouseId, setWarehouseId] = useState<string>(searchParams.get('warehouseId') ?? '')
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null)
  const [dispatchResult, setDispatchResult] = useState<ApiDispatchResponse | null>(null)
  const [dispatchLoading, setDispatchLoading] = useState(false)
  const [dispatchError, setDispatchError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [sidebarWidth, setSidebarWidth] = useState(240)
  const [costPanelWidth, setCostPanelWidth] = useState(720)
  const [readyDirty, setReadyDirty] = useState(false)
  const [metricDetail, setMetricDetail] = useState<'p_cover' | 'fill_rate' | 'cpo' | 'fleet_utilization' | 'capacity_shortfall' | null>(null)

  // AbortController for cancelling in-flight dispatch requests
  const dispatchAbortRef = useRef<AbortController | null>(null)
  // Mutable ref tracking the current warehouseId for mass-refresh callback
  const warehouseIdRef = useRef(warehouseId)
  warehouseIdRef.current = warehouseId

  // Effective loading: single-warehouse dispatch OR mass-refresh without data yet
  const showLoading = dispatchLoading || (refreshingAll && !dispatchResult)

  // ── Forecast cache: localStorage-backed, keyed by `wid__datetime` ────────
  const LS_PREFIX = 'dispatch_cache__'
  const cacheKey = (wid: string, dt: string) => `${wid}__${dt}`

  function lsGet(key: string): ApiDispatchResponse | null {
    try {
      const raw = localStorage.getItem(LS_PREFIX + key)
      return raw ? (JSON.parse(raw) as ApiDispatchResponse) : null
    } catch { return null }
  }
  function lsSet(key: string, val: ApiDispatchResponse) {
    try { localStorage.setItem(LS_PREFIX + key, JSON.stringify(val)) } catch { /* quota */ }
  }
  function lsDel(key: string) {
    try { localStorage.removeItem(LS_PREFIX + key) } catch { /* ignore */ }
  }

  const startDragSidebar = useCallback((e: ReactMouseEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = sidebarWidth
    const onMove = (ev: MouseEvent) => setSidebarWidth(Math.max(160, Math.min(640, startW + ev.clientX - startX)))
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [sidebarWidth])

  const startDragCost = useCallback((e: ReactMouseEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = costPanelWidth
    const onMove = (ev: MouseEvent) => setCostPanelWidth(Math.max(280, Math.min(900, startW - (ev.clientX - startX))))
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [costPanelWidth])

  const selectedWarehouse = warehouses.find(w => w.id === warehouseId) ?? null
  const warehouseRoutes = selectedWarehouse ? routes.filter(r => r.fromId === selectedWarehouse.id) : []

  // ── Dispatch logic ───────────────────────────────────────────────────────
  function deriveWarehouseStatus(result: ApiDispatchResponse): 'ok' | 'warning' | 'critical' {
    const shortfall = result.metrics?.fleet_capacity_shortfall ?? 0
    const hasShortfall = shortfall > 0
    let hasFullyMissed = false
    let hasPartiallyMissed = false
    for (const rp of result.routes) {
      for (const row of rp.plan) {
        if (row.demand_new > 0 && row.vehicles_count === 0) { hasFullyMissed = true }
        else if (row.leftover_stock >= 1) { hasPartiallyMissed = true }
      }
    }
    if (hasShortfall && hasFullyMissed) return 'critical'
    if (hasShortfall && hasPartiallyMissed) return 'warning'
    return 'ok'
  }

  const runDispatch = useCallback(async (wid: string, forceRefresh = false) => {
    if (!wid) return
    const key = cacheKey(wid, analysisDateTime)
    if (!forceRefresh) {
      const cached = lsGet(key)
      if (cached) {
        setDispatchResult(cached)
        setDispatchError(null)
        setWarehouseStatus(wid, deriveWarehouseStatus(cached))
        return
      }
    }
    // Cancel previous in-flight request
    dispatchAbortRef.current?.abort()
    const controller = new AbortController()
    dispatchAbortRef.current = controller

    setDispatchLoading(true)
    setDispatchError(null)
    setDispatchResult(null)
    try {
      const ts = analysisDateTime.replace('T', ' ') + ':00'
      const result = await postDispatch({
        warehouse_id: wid,
        timestamp: ts,
        incoming_vehicles: incomingVehicles.length > 0 ? incomingVehicles : undefined,
        granularity: riskSettings.granularity,
      }, controller.signal)
      if (controller.signal.aborted) return
      lsSet(key, result)
      setDispatchResult(result)
      setWarehouseStatus(wid, deriveWarehouseStatus(result))
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      setDispatchError(err instanceof Error ? err.message : String(err))
      setDispatchResult(null)
    } finally {
      if (!controller.signal.aborted) setDispatchLoading(false)
    }
  }, [analysisDateTime, incomingVehicles, setWarehouseStatus, riskSettings.granularity])

  const runDispatchRef = useRef(runDispatch)
  useEffect(() => { runDispatchRef.current = runDispatch }, [runDispatch])

  // On initial mount: if warehouseId came from URL param, load per-warehouse data and run dispatch
  useEffect(() => {
    if (warehouseId) {
      Promise.all([getVehicles(warehouseId), getIncomingVehicles(warehouseId)])
        .then(([vList, iRes]) => {
          setVehicleTypes(vList.map(apiVehicleToVehicleType))
          setIncomingVehicles(iRes.incoming ?? [])
        })
        .catch(() => {})
      runDispatchRef.current(warehouseId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // intentionally mount-only

  // During mass refresh, pick up results from LS cache as each warehouse completes
  useEffect(() => {
    if (!refreshingAll || !warehouseId) return
    if (dispatchResult) return // already have data
    const cached = lsGet(cacheKey(warehouseId, analysisDateTime))
    if (cached) {
      setDispatchResult(cached)
    }
  }, [refreshingAll, warehouseId, warehouseStatuses, analysisDateTime, dispatchResult])

  // Auto-dispatch when warehouse is selected
  const handleSelectWarehouse = useCallback(async (id: string) => {
    // If same warehouse is already loading — ignore the click to avoid aborting + 409
    if (id === warehouseId && dispatchLoading) return

    // Cancel any in-flight single dispatch for a DIFFERENT warehouse
    if (id !== warehouseId) {
      dispatchAbortRef.current?.abort()
      setDispatchLoading(false)
    }
    setWarehouseId(id)
    setSelectedWarehouseId(id)
    setSelectedRouteId(null)
    setDispatchError(null)
    // Try LS cache first (may be populated by mass refresh)
    const cached = lsGet(cacheKey(id, analysisDateTime))
    if (cached) {
      setDispatchResult(cached)
      setWarehouseStatus(id, deriveWarehouseStatus(cached))
    } else {
      setDispatchResult(null)
    }
    // Reload vehicles and incoming for the selected warehouse
    try {
      const [vList, iRes] = await Promise.all([getVehicles(id), getIncomingVehicles(id)])
      setVehicleTypes(vList.map(apiVehicleToVehicleType))
      setIncomingVehicles(iRes.incoming ?? [])
    } catch { /* ignore */ }
    // Only dispatch if NOT in mass refresh — mass refresh will produce the result
    if (!refreshingAllRef.current) {
      runDispatchRef.current(id)
    }
  }, [warehouseId, dispatchLoading, setSelectedWarehouseId, setVehicleTypes, setIncomingVehicles, analysisDateTime, setWarehouseStatus])

  const handleRefresh = useCallback(() => {
    if (!warehouseId) return
    // Invalidate localStorage cache and clear result immediately so loading shows
    lsDel(cacheKey(warehouseId, analysisDateTime))
    setDispatchResult(null)
    runDispatchRef.current(warehouseId, true)
  }, [warehouseId, analysisDateTime])

  useEffect(() => {
    if (warehouseRoutes.length === 0) {
      setSelectedRouteId(null)
      return
    }
    setSelectedRouteId(prev => (prev && warehouseRoutes.some(route => route.id === prev) ? prev : warehouseRoutes[0].id))
  }, [warehouseRoutes])

  // Sync URL param → auto-dispatch when navigated from map drawer
  useEffect(() => {
    const id = searchParams.get('warehouseId') ?? ''
    if (id && id !== warehouseId) {
      handleSelectWarehouse(id)
    }
  }, [searchParams]) // eslint-disable-line react-hooks/exhaustive-deps

  const STATUS_COLOR: Record<string, string> = {
    none: 'bg-accent/60',
    ok: 'bg-status-green',
    warning: 'bg-status-yellow',
    critical: 'bg-status-red',
  }

  const STATUS_ORDER: Record<string, number> = { critical: 0, warning: 1, ok: 2, none: 3 }
  const filteredWarehouses = warehouses
    .filter(w =>
      w.name.toLowerCase().includes(search.toLowerCase()) ||
      w.city.toLowerCase().includes(search.toLowerCase()),
    )
    .sort((a, b) => {
      const sa = STATUS_ORDER[warehouseStatuses[a.id] ?? 'none'] ?? 3
      const sb = STATUS_ORDER[warehouseStatuses[b.id] ?? 'none'] ?? 3
      return sa - sb
    })

  const selectedRoute = warehouseRoutes.find(route => route.id === selectedRouteId) ?? null
  const selectedRoutePlan = dispatchResult?.routes.find(route => route.route_id === selectedRouteId) ?? null
  const handleUpdateRouteReadyToShip = useCallback(async (routeId: string, value: number) => {
    const nextRoutes = routes.map(route => route.id === routeId ? { ...route, readyToShip: value } : route)
    setRoutes(nextRoutes)
    try {
      await putRouteDistances(nextRoutes.map(routeDistanceToApi))
      if (warehouseId) {
        runDispatchRef.current(warehouseId)
      }
    } catch (err) {
      setDispatchError(err instanceof Error ? err.message : String(err))
    }
  }, [routes, setRoutes, warehouseId])

  const handleFleetChange = useCallback(async (
    vehicleType: string,
    horizonIdx: number,
    newCount: number,
  ) => {
    if (horizonIdx === 0) {
      // Update base available count on the vehicle record for current warehouse only
      const vt = vehicleTypes.find(v => v.id === vehicleType)
      if (!vt) return
      await updateVehicle(vehicleType, {
        vehicle_type: vehicleType,
        capacity_units: vt.capacity,
        cost_per_km: vt.costPerKm,
        available: newCount,
        category: vt.category,
        underload_penalty: vt.underloadPenalty,
        fixed_dispatch_cost: vt.fixedDispatchCost,
        warehouse_id: warehouseId || undefined,
      })
      const reloaded = await getVehicles(warehouseId || undefined)
      const newTypes = reloaded.map(apiVehicleToVehicleType)
      setVehicleTypes(newTypes)
    } else {
      // Arrivals at this exact horizon = newCount − totalAvailableAtPreviousHorizon
      const vt = vehicleTypes.find(v => v.id === vehicleType)
      const base = vt?.available ?? 0
      const additionsBelow = incomingVehicles
        .filter(iv => iv.vehicle_type === vehicleType && iv.horizon_idx < horizonIdx)
        .reduce((s, iv) => s + iv.count, 0)
      const prevHorizTotal = base + additionsBelow
      const delta = newCount - prevHorizTotal
      // Replace all existing entries for this type+horizon with a single entry (or remove)
      const filtered = incomingVehicles.filter(
        iv => !(iv.vehicle_type === vehicleType && iv.horizon_idx === horizonIdx)
      )
      const newList = delta > 0
        ? [...filtered, { vehicle_type: vehicleType, horizon_idx: horizonIdx, count: delta }]
        : filtered
      await putIncomingVehicles(newList, warehouseId || undefined)
      setIncomingVehicles(newList)
    }
    // Invalidate cache and re-run dispatch; reload both vehicles + incoming
    if (warehouseId) {
      lsDel(cacheKey(warehouseId, analysisDateTime))
      const [vList, iRes] = await Promise.all([getVehicles(warehouseId), getIncomingVehicles(warehouseId)])
      setVehicleTypes(vList.map(apiVehicleToVehicleType))
      setIncomingVehicles(iRes.incoming ?? [])
      await runDispatchRef.current(warehouseId, true)
    }
  }, [vehicleTypes, incomingVehicles, warehouseId, analysisDateTime, setVehicleTypes, setIncomingVehicles])

  const handleSyncVehicle = useCallback(async (vehicleType: string) => {
    if (!warehouseId) return
    await syncVehicleAcrossWarehouses(vehicleType, warehouseId)
  }, [warehouseId])

  const handleCallRoute = useCallback(async (routeId: string) => {
    const ts = analysisDateTime.replace('T', ' ') + ':00'
    const res = await callRoute({ route_id: routeId, timestamp: ts, warehouse_id: warehouseId || undefined })
    return JSON.stringify(res.request, null, 2)
  }, [analysisDateTime, warehouseId])

  const handleCallAllRoutes = useCallback(async (routeIds: string[]): Promise<void> => {
    const ts = analysisDateTime.replace('T', ' ') + ':00'
    const payloads = []
    for (const rid of routeIds) {
      const res = await callRoute({ route_id: rid, timestamp: ts, warehouse_id: warehouseId || undefined })
      payloads.push(res.request)
    }
    const json = JSON.stringify({ routes: payloads }, null, 2)
    const blob = new Blob([json], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const w = window.open(url, '_blank', 'noopener,noreferrer')
    w?.blur?.()
    window.focus()
    setTimeout(() => URL.revokeObjectURL(url), 5000)
  }, [handleCallRoute])

  return (
    <div className="flex flex-col h-full overflow-hidden relative">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-surface shrink-0 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-base font-semibold text-foreground">Оптимизатор транспортных вызовов</h1>
          <p className="text-xs text-muted mt-0.5">
            Выберите склад — прогноз загрузится автоматически.
          </p>
        </div>
        {warehouseId && (
          <button
            onClick={handleRefresh}
            disabled={dispatchLoading || refreshingAll}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium border border-border transition-colors disabled:opacity-50 shrink-0",
              readyDirty && !dispatchLoading
                ? "bg-accent text-background animate-[pulse_1s_ease-in-out_infinite]"
                : "bg-elevated text-foreground hover:border-accent hover:text-accent"
            )}
          >
            <RefreshCw className={cn('w-3.5 h-3.5', dispatchLoading && 'animate-spin')} />
            Обновить прогноз
          </button>
        )}
        <button
          onClick={async () => {
            // Cancel any in-flight single dispatch
            dispatchAbortRef.current?.abort()
            setDispatchLoading(false)
            setDispatchResult(null)
            setDispatchError(null)
            // refreshAllWarehouses sets refreshingAll=true → showLoading activates
            // When each warehouse completes, callback updates local result via mutable ref
            await refreshAllWarehouses((wid, result) => {
              if (wid === warehouseIdRef.current) {
                setDispatchResult(result)
              }
            })
          }}
          disabled={refreshingAll || dispatchLoading}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium border border-border transition-colors disabled:opacity-50 shrink-0',
            'bg-elevated text-foreground hover:border-accent hover:text-accent',
          )}
        >
          <RefreshCw className={cn('w-3.5 h-3.5', refreshingAll && 'animate-spin')} />
          Обновить все
        </button>
      </div>

      {/* P_cover + metrics strip — removed; P-cover moved to Бизнес-метрики block below */}

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* ── Warehouse sidebar ──────────────────────────────────────────── */}
        <div style={{ width: sidebarWidth }} className="shrink-0 flex flex-col overflow-hidden bg-surface border-r border-border">
          <div className="px-3 py-2 border-b border-border">
            <input
              type="text"
              placeholder="Поиск склада…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full bg-elevated border border-border rounded px-2 py-1.5 text-xs text-foreground focus:outline-none focus:border-accent"
            />
          </div>
          <div className="flex-1 overflow-y-auto">
            {filteredWarehouses.map(w => (
              <button
                key={w.id}
                onClick={() => handleSelectWarehouse(w.id)}
                className={cn(
                  'w-full text-left px-3 py-2.5 transition-colors border-l-2',
                  warehouseId === w.id
                    ? 'bg-accent/10 border-accent'
                    : 'hover:bg-elevated border-transparent',
                )}
              >
                <div className="flex items-center gap-2">
                  {refreshingAll && !warehouseStatuses[w.id] ? (
                    <RefreshCw className="w-3 h-3 animate-spin text-accent shrink-0" />
                  ) : (
                    <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', STATUS_COLOR[warehouseStatuses[w.id] ?? 'none'])} />
                  )}
                  <span className={cn('font-medium truncate text-xs', warehouseId === w.id ? 'text-accent' : 'text-foreground')}>
                    {w.name}
                  </span>
                </div>
                <div className="text-[11px] text-muted ml-3.5">{w.city}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Sidebar resize handle */}
        <div onMouseDown={startDragSidebar} className="w-1 shrink-0 cursor-col-resize hover:bg-accent/50 transition-colors" />

        {/* ── Main content ───────────────────────────────────────────────── */}
        <div className="flex-1 flex overflow-hidden">
          {/* Route table */}
          <div className="flex-1 min-w-0 overflow-y-auto p-4">
            {!warehouseId ? (
              <div className="flex items-center justify-center h-full text-muted text-sm">
                Выберите склад слева для просмотра маршрутов и прогноза
              </div>
            ) : (
              <div className="space-y-4">
                {/* ── Бизнес-метрики ──────────────────────────────────────── */}
                {(dispatchResult?.metrics || showLoading) && (() => {
                  if (showLoading && !dispatchResult?.metrics) {
                    return (
                      <div className="bg-surface border border-border rounded-lg px-4 py-3">
                        <div className="section-label mb-3">Бизнес-метрики</div>
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 animate-pulse">
                          {[...Array(5)].map((_, i) => (
                            <div key={i} className="flex flex-col gap-1.5">
                              <div className="h-3 bg-elevated rounded w-3/4" />
                              <div className="h-7 bg-elevated rounded w-1/2" />
                              <div className="h-2.5 bg-elevated rounded w-2/3" />
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  }
                  const m = dispatchResult!.metrics!
                  const horizonLabels = m.horizon_labels ?? makeHorizonLabels(riskSettings.granularity)
                  const pColor = m.p_cover >= 0.9 ? 'text-status-green' : m.p_cover >= 0.7 ? 'text-status-yellow' : 'text-status-red'
                  const utilRatio = m.fleet_utilization_ratio
                  const utilColor = utilRatio == null ? 'text-muted'
                    : utilRatio <= 0.8 ? 'text-status-green'
                    : utilRatio <= 1.0 ? 'text-status-yellow'
                    : 'text-status-red'
                  const shortfall = m.fleet_capacity_shortfall
                  const shortColor = shortfall == null ? 'text-muted'
                    : shortfall <= 0 ? 'text-status-green'
                    : shortfall <= 50 ? 'text-status-yellow'
                    : 'text-status-red'
                  return (
                    <div className={`bg-surface border border-border rounded-lg px-4 py-3 transition-opacity ${showLoading ? 'opacity-50' : ''}`}>
                      <div className="section-label mb-3">Бизнес-метрики <span className="text-[10px] text-muted font-normal ml-1">(клик → детализация)</span></div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                        {/* P покрытия */}
                        <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('p_cover')}>
                          <span className="text-[11px] text-muted leading-tight">Вероятность покрытия спроса</span>
                          <span className={`text-xl font-bold font-mono ${pColor}`}>{(m.p_cover * 100).toFixed(1)}%</span>
                          <div className="flex flex-wrap gap-x-2 gap-y-0.5 mt-0.5">
                            {m.p_cover_by_horizon.map((p, i) => i === 0 ? null : (
                              <span key={i} className="text-[10px] text-muted">
                                {horizonDisplayLabel(horizonLabels[i])}&nbsp;<span className={p >= 0.9 ? 'text-status-green' : p >= 0.7 ? 'text-status-yellow' : 'text-status-red'}>{(p*100).toFixed(0)}%</span>
                              </span>
                            ))}
                          </div>
                        </div>
                        {/* Fill Rate */}
                        <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('fill_rate')}>
                          <span className="text-[11px] text-muted leading-tight">Коэффициент загрузки ТС</span>
                          <span className={`text-xl font-bold font-mono ${m.fill_rate >= 0.8 ? 'text-status-green' : m.fill_rate >= 0.5 ? 'text-status-yellow' : 'text-status-red'}`}>
                            {(m.fill_rate * 100).toFixed(1)}%
                          </span>
                          <span className="text-[10px] text-muted">Отправлено / вместимость</span>
                        </div>
                        {/* CPO */}
                        <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('cpo')}>
                          <span className="text-[11px] text-muted leading-tight">Стоимость одной доставки</span>
                          <span className="text-xl font-bold font-mono text-accent">{m.cpo.toFixed(0)} ₽</span>
                          <span className="text-[10px] text-muted">Общие затраты / ед. отправлено</span>
                        </div>
                        {/* Fleet utilization ratio */}
                        <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('fleet_utilization')}>
                          <span className="text-[11px] text-muted leading-tight">Коэффициент утилизации флота</span>
                          <span className={`text-xl font-bold font-mono ${utilColor}`}>
                            {utilRatio != null ? utilRatio.toFixed(2) : '—'}
                          </span>
                          <span className="text-[10px] text-muted">
                            {m.required_capacity_units != null && m.available_capacity_units != null
                              ? `${Math.round(m.required_capacity_units)} / ${Math.round(m.available_capacity_units)} ед.`
                              : 'Треб. / доступно'}
                          </span>
                        </div>
                        {/* Capacity shortfall */}
                        <div className="flex flex-col gap-0.5 cursor-pointer hover:bg-white/5 rounded-lg p-1.5 -m-1.5 transition-colors" onClick={() => setMetricDetail('capacity_shortfall')}>
                          <span className="text-[11px] text-muted leading-tight">Нехватка вместимости</span>
                          <span className={`text-xl font-bold font-mono ${shortColor}`}>
                            {shortfall != null
                              ? (shortfall > 0 ? `+${Math.round(shortfall)}` : Math.round(shortfall).toString())
                              : '—'} ед.
                          </span>
                          <span className="text-[10px] text-muted">Треб. − доступно</span>
                        </div>
                      </div>
                      <MetricDetailModal metricKey={metricDetail} metrics={m} onClose={() => setMetricDetail(null)} />
                    </div>
                  )
                })()}
                <RouteTable
                  warehouse={selectedWarehouse}
                  warehouseRoutes={warehouseRoutes}
                  dispatchResult={dispatchResult}
                  loading={showLoading}
                  error={dispatchError}
                  selectedRouteId={selectedRouteId}
                  onSelectRoute={setSelectedRouteId}
                  onChangeReadyToShip={handleUpdateRouteReadyToShip}
                  onCallRoute={handleCallRoute}
                  onCallAllRoutes={handleCallAllRoutes}
                  vehicleTypes={vehicleTypes}
                  incomingVehicles={incomingVehicles}
                  onFleetChange={handleFleetChange}
                  onSyncVehicle={handleSyncVehicle}
                  onReadyDirtyChange={setReadyDirty}
                  analysisDateTime={analysisDateTime}
                  granularity={riskSettings.granularity}
                />
              </div>
            )}
          </div>

          {/* Cost panel resize handle */}
          <div onMouseDown={startDragCost} className="w-1 shrink-0 cursor-col-resize hover:bg-accent/50 transition-colors border-l border-border" />

          {/* Cost scenarios */}
          <div style={{ width: costPanelWidth }} className="shrink-0 overflow-y-auto p-4">
            <CostBenefitCard
              route={selectedRoute}
              routePlan={selectedRoutePlan}
              vehicleTypes={selectedWarehouse?.vehicles ?? []}
              riskSettings={riskSettings}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
