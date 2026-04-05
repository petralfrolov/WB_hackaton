import { useState, useCallback, useEffect, useRef, type MouseEvent as ReactMouseEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { ApiDispatchResponse } from '../types'
import { cn, routeDistanceToApi, makeHorizonLabels, horizonDisplayLabel } from '../lib/utils'
import { RouteTable } from '../components/optimizer/RouteTable'
import { CostBenefitCard } from '../components/optimizer/CostBenefitCard'
import { useSimulationContext } from '../context/SimulationContext'
import { postDispatch, putRouteDistances, updateVehicle, getVehicles, putIncomingVehicles, callRoute } from '../api'
import { apiVehicleToVehicleType } from '../lib/utils'
import { RefreshCw } from 'lucide-react'

export function OptimizerPage() {
  const [searchParams] = useSearchParams()
  const { warehouses, routes, setRoutes, riskSettings, analysisDateTime, setSelectedWarehouseId, incomingVehicles, vehicleTypes, setVehicleTypes, setIncomingVehicles, warehouseStatuses, setWarehouseStatus, refreshAllWarehouses, refreshingAll } = useSimulationContext()

  const [warehouseId, setWarehouseId] = useState<string>(searchParams.get('warehouseId') ?? '')
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null)
  const [dispatchResult, setDispatchResult] = useState<ApiDispatchResponse | null>(null)
  const [dispatchLoading, setDispatchLoading] = useState(false)
  const [dispatchError, setDispatchError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [sidebarWidth, setSidebarWidth] = useState(240)
  const [costPanelWidth, setCostPanelWidth] = useState(720)
  const [readyDirty, setReadyDirty] = useState(false)

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
    let hasCritical = false
    let hasWarning = false
    for (const rp of result.routes) {
      for (const row of rp.plan) {
        if (row.demand_new > 0 && row.vehicles_count === 0) { hasCritical = true }
        else if (row.leftover_stock >= 1) { hasWarning = true }
      }
    }
    if (hasCritical) return 'critical'
    if (hasWarning) return 'warning'
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
    setDispatchLoading(true)
    setDispatchError(null)
    try {
      const ts = analysisDateTime.replace('T', ' ') + ':00'
      const result = await postDispatch({
        warehouse_id: wid,
        timestamp: ts,
        incoming_vehicles: incomingVehicles.length > 0 ? incomingVehicles : undefined,
        granularity: riskSettings.granularity,
      })
      lsSet(key, result)
      setDispatchResult(result)
      setWarehouseStatus(wid, deriveWarehouseStatus(result))
    } catch (err) {
      setDispatchError(err instanceof Error ? err.message : String(err))
      setDispatchResult(null)
    } finally {
      setDispatchLoading(false)
    }
  }, [analysisDateTime, incomingVehicles, setWarehouseStatus, riskSettings.granularity])

  const runDispatchRef = useRef(runDispatch)
  useEffect(() => { runDispatchRef.current = runDispatch }, [runDispatch])

  // On initial mount: if warehouseId came from URL param, run dispatch (hits LS cache or fetches)
  useEffect(() => {
    if (warehouseId) runDispatchRef.current(warehouseId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // intentionally mount-only

  // Auto-dispatch when warehouse is selected
  const handleSelectWarehouse = useCallback((id: string) => {
    setWarehouseId(id)
    setSelectedWarehouseId(id)
    setSelectedRouteId(null)
    // load from cache or fetch
    runDispatchRef.current(id)
  }, [setSelectedWarehouseId])

  const handleRefresh = useCallback(() => {
    if (!warehouseId) return
    // Invalidate localStorage cache for current key and re-fetch
    lsDel(cacheKey(warehouseId, analysisDateTime))
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

  const filteredWarehouses = warehouses.filter(w =>
    w.name.toLowerCase().includes(search.toLowerCase()) ||
    w.city.toLowerCase().includes(search.toLowerCase()),
  )

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
      // Update base available count on the vehicle record
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
      })
      const reloaded = await getVehicles()
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
      await putIncomingVehicles(newList)
      setIncomingVehicles(newList)
    }
    // Invalidate cache and re-run dispatch
    if (warehouseId) {
      lsDel(cacheKey(warehouseId, analysisDateTime))
      await runDispatchRef.current(warehouseId, true)
    }
  }, [vehicleTypes, incomingVehicles, warehouseId, analysisDateTime, setVehicleTypes, setIncomingVehicles])

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
      {/* Blur overlay while refreshing all warehouses */}
      {refreshingAll && (
        <div className="absolute inset-0 z-50 backdrop-blur-sm bg-background/60 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3 bg-surface border border-border rounded-xl px-8 py-6 shadow-lg">
            <RefreshCw className="w-8 h-8 animate-spin text-accent" />
            <p className="text-sm text-foreground font-medium">Обновление всех складов…</p>
            <p className="text-xs text-muted">Это может занять несколько секунд</p>
          </div>
        </div>
      )}
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
          onClick={async () => { await refreshAllWarehouses(); if (warehouseId) runDispatchRef.current(warehouseId) }}
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

      {/* P_cover + metrics strip */}
      {dispatchResult?.metrics && (() => {
        const m = dispatchResult.metrics
        const pColor = m.p_cover >= 0.9 ? 'text-green-400' : m.p_cover >= 0.7 ? 'text-yellow-400' : 'text-red-400'
        const horizonLabels = m.horizon_labels ?? makeHorizonLabels(riskSettings.granularity)
        return (
          <div className="px-4 py-2 border-b border-border bg-elevated flex items-center gap-6 text-xs shrink-0 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-muted">P(хватит транспорта):</span>
              <span className={`font-bold text-sm ${pColor}`}>{(m.p_cover * 100).toFixed(1)}%</span>
              <span className="text-muted hidden sm:inline">
                ({m.p_cover_by_horizon.map((p, i) => i === 0 ? null : (
                  <span key={i}>{horizonDisplayLabel(horizonLabels[i])}&nbsp;<span className={p >= 0.9 ? 'text-green-400' : p >= 0.7 ? 'text-yellow-400' : 'text-red-400'}>{(p*100).toFixed(0)}%</span>{i < m.p_cover_by_horizon.length - 1 ? ' · ' : ''}</span>
                ))})
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted">Fill Rate:</span>
              <span className="font-semibold text-foreground">{(m.fill_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted">CPO:</span>
              <span className="font-semibold text-foreground">{m.cpo.toFixed(0)} ₽/ед.</span>
            </div>
          </div>
        )
      })()}

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
                  <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', STATUS_COLOR[warehouseStatuses[w.id] ?? 'none'])} />
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
              <RouteTable
                warehouse={selectedWarehouse}
                warehouseRoutes={warehouseRoutes}
                dispatchResult={dispatchResult}
                loading={dispatchLoading}
                error={dispatchError}
                selectedRouteId={selectedRouteId}
                onSelectRoute={setSelectedRouteId}
                onChangeReadyToShip={handleUpdateRouteReadyToShip}
                onCallRoute={handleCallRoute}
                onCallAllRoutes={handleCallAllRoutes}
                vehicleTypes={vehicleTypes}
                incomingVehicles={incomingVehicles}
                onFleetChange={handleFleetChange}
                onReadyDirtyChange={setReadyDirty}
                analysisDateTime={analysisDateTime}
                granularity={riskSettings.granularity}
              />
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
