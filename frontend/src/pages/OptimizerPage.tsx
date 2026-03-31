import { useState, useCallback, useEffect, useRef, type MouseEvent as ReactMouseEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { ApiDispatchResponse } from '../types'
import { cn, routeDistanceToApi } from '../lib/utils'
import { RouteTable } from '../components/optimizer/RouteTable'
import { CostBenefitCard } from '../components/optimizer/CostBenefitCard'
import { useSimulationContext } from '../context/SimulationContext'
import { postDispatch, putRouteDistances, updateVehicle, getVehicles, putIncomingVehicles } from '../api'
import { apiVehicleToVehicleType } from '../lib/utils'
import { RefreshCw } from 'lucide-react'

export function OptimizerPage() {
  const [searchParams] = useSearchParams()
  const { warehouses, routes, setRoutes, riskSettings, analysisDateTime, setSelectedWarehouseId, incomingVehicles, vehicleTypes, setVehicleTypes, setIncomingVehicles } = useSimulationContext()

  const [warehouseId, setWarehouseId] = useState<string>(searchParams.get('warehouseId') ?? '')
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null)
  const [dispatchResult, setDispatchResult] = useState<ApiDispatchResponse | null>(null)
  const [dispatchLoading, setDispatchLoading] = useState(false)
  const [dispatchError, setDispatchError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [sidebarWidth, setSidebarWidth] = useState(240)
  const [costPanelWidth, setCostPanelWidth] = useState(720)

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
  const runDispatch = useCallback(async (wid: string, forceRefresh = false) => {
    if (!wid) return
    const key = cacheKey(wid, analysisDateTime)
    if (!forceRefresh) {
      const cached = lsGet(key)
      if (cached) {
        setDispatchResult(cached)
        setDispatchError(null)
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
      })
      lsSet(key, result)
      setDispatchResult(result)
    } catch (err) {
      setDispatchError(err instanceof Error ? err.message : String(err))
      setDispatchResult(null)
    } finally {
      setDispatchLoading(false)
    }
  }, [analysisDateTime, incomingVehicles])

  const runDispatchRef = useRef(runDispatch)
  useEffect(() => { runDispatchRef.current = runDispatch }, [runDispatch])

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
    horizonIdx: 0 | 1 | 2 | 3,
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
        ? [...filtered, { vehicle_type: vehicleType, horizon_idx: horizonIdx as 0|1|2|3, count: delta }]
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

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-surface shrink-0 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-base font-semibold text-foreground">Оптимизатор транспортных вызовов</h1>
          <p className="text-xs text-muted mt-0.5">
            Выберите склад — прогноз загрузится автоматически (кэшируется до ручного обновления).
          </p>
        </div>
        {warehouseId && (
          <button
            onClick={handleRefresh}
            disabled={dispatchLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-elevated border border-border text-foreground hover:border-accent hover:text-accent transition-colors disabled:opacity-50 shrink-0"
          >
            <RefreshCw className={cn('w-3.5 h-3.5', dispatchLoading && 'animate-spin')} />
            Обновить прогноз
          </button>
        )}
      </div>

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
                  <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', STATUS_COLOR[w.status])} />
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
                vehicleTypes={vehicleTypes}
                incomingVehicles={incomingVehicles}
                onFleetChange={handleFleetChange}
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
