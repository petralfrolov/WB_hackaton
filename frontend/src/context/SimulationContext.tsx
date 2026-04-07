import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import type { ApiDispatchResponse, ApiIncomingVehicle, ApiWarehouseMetrics, RouteDistance, RiskSettings, VehicleType, Warehouse } from '../types'
import { getWarehouses, getRouteDistances, getConfig, getVehicles, getIncomingVehicles, patchSettings, postDispatch } from '../api'
import {
  defaultRiskSettings,
  apiWarehouseToWarehouse,
  apiRouteDistanceToRouteDistance,
  apiVehicleToVehicleType,
  makeSankey,
} from '../lib/utils'

interface SimulationContextValue {
  warehouses: Warehouse[]
  routes: RouteDistance[]
  setRoutes: (routes: RouteDistance[]) => void
  vehicleTypes: VehicleType[]
  setVehicleTypes: (types: VehicleType[]) => void
  riskSettings: RiskSettings
  setRiskSettings: (settings: RiskSettings) => void
  analysisDateTime: string
  setAnalysisDateTime: (value: string) => void
  selectedWarehouseId: string | null
  setSelectedWarehouseId: (id: string | null) => void
  incomingVehicles: ApiIncomingVehicle[]
  setIncomingVehicles: (list: ApiIncomingVehicle[]) => void
  warehouseStatuses: Record<string, 'none' | 'ok' | 'warning' | 'critical'>
  setWarehouseStatus: (id: string, status: 'none' | 'ok' | 'warning' | 'critical') => void
  warehouseMetrics: Record<string, ApiWarehouseMetrics>
  setWarehouseMetrics: (id: string, metrics: ApiWarehouseMetrics) => void
  clearCache: () => void
  refreshAllWarehouses: (onWarehouseComplete?: (warehouseId: string, result: ApiDispatchResponse) => void) => Promise<void>
  refreshingAll: boolean
}

const SimulationContext = createContext<SimulationContextValue | null>(null)

function applyRouteTotalsToWarehouses(warehouses: Warehouse[], routes: RouteDistance[]): Warehouse[] {
  return warehouses.map(warehouse => {
    const totalReady = routes
      .filter(route => route.fromId === warehouse.id)
      .reduce((sum, route) => sum + route.readyToShip, 0)

    return {
      ...warehouse,
      readyToShip: totalReady,
      sankeyData: makeSankey(totalReady * 4),
    }
  })
}

function toDateTimeLocalValue(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`
}

function normalizeToHalfHour(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value

  const minutes = date.getMinutes()
  const normalizedMinutes = minutes < 15 ? 0 : minutes < 45 ? 30 : 60

  if (normalizedMinutes === 60) {
    date.setHours(date.getHours() + 1)
    date.setMinutes(0, 0, 0)
  } else {
    date.setMinutes(normalizedMinutes, 0, 0)
  }

  return toDateTimeLocalValue(date)
}

export function SimulationProvider({ children }: { children: ReactNode }) {
  const [warehouses, setWarehouses] = useState<Warehouse[]>([])
  const [routes, setRoutes] = useState<RouteDistance[]>([])
  const [vehicleTypes, setVehicleTypes] = useState<VehicleType[]>([])
  const [riskSettings, setRiskSettingsState] = useState<RiskSettings>(defaultRiskSettings)
  const [analysisDateTime, setAnalysisDateTime] = useState<string>(
    normalizeToHalfHour(localStorage.getItem('wb_analysis_datetime') ?? '2025-03-13T11:00'),
  )
  const [refreshingAll, setRefreshingAll] = useState(false)
  const refreshingAllRef = useRef(false)
  const [selectedWarehouseId, setSelectedWarehouseId] = useState<string | null>(null)
  const [incomingVehicles, setIncomingVehicles] = useState<ApiIncomingVehicle[]>([])
  const [warehouseStatuses, setWarehouseStatuses] = useState<Record<string, 'none' | 'ok' | 'warning' | 'critical'>>({});
  const [warehouseMetrics, setWarehouseMetricsState] = useState<Record<string, ApiWarehouseMetrics>>({});

  const setWarehouseMetrics = useCallback((id: string, metrics: ApiWarehouseMetrics) => {
    setWarehouseMetricsState(prev => ({ ...prev, [id]: metrics }))
  }, [])

  const setWarehouseStatus = useCallback((id: string, status: 'none' | 'ok' | 'warning' | 'critical') => {
    setWarehouseStatuses(prev => prev[id] === status ? prev : { ...prev, [id]: status })
  }, [])

  // Persist last selected datetime to localStorage
  useEffect(() => {
    localStorage.setItem('wb_analysis_datetime', analysisDateTime)
  }, [analysisDateTime])

  // On analysisDateTime change: reset statuses then restore from LS cache for new datetime
  useEffect(() => {
    setWarehouseStatuses({})
    const prefix = 'dispatch_cache__'
    const suffix = `__${analysisDateTime}`
    const statuses: Record<string, 'ok' | 'warning' | 'critical'> = {}
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (!k?.startsWith(prefix)) continue
      const inner = k.slice(prefix.length) // e.g. "wh1__2025-03-13T11:00"
      if (!inner.endsWith(suffix)) continue
      const warehouseId = inner.slice(0, inner.length - suffix.length)
      try {
        const cached = JSON.parse(localStorage.getItem(k)!)
        let hasCritical = false, hasWarning = false
        for (const rp of cached.routes ?? []) {
          for (const row of rp.plan ?? []) {
            if (row.demand_new > 0 && row.vehicles_count === 0) hasCritical = true
            else if (row.leftover_stock >= 1) hasWarning = true
          }
        }
        statuses[warehouseId] = hasCritical ? 'critical' : hasWarning ? 'warning' : 'ok'
      } catch { /* corrupt entry, skip */ }
    }
    setWarehouseStatuses(statuses)
    // Restore metrics from LS cache for new datetime
    const newMetrics: Record<string, ApiWarehouseMetrics> = {}
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (!k?.startsWith(prefix)) continue
      const inner = k.slice(prefix.length)
      if (!inner.endsWith(suffix)) continue
      const warehouseId = inner.slice(0, inner.length - suffix.length)
      try {
        const cached = JSON.parse(localStorage.getItem(k)!)
        if (cached.metrics) newMetrics[warehouseId] = cached.metrics
      } catch { /* corrupt */ }
    }
    setWarehouseMetricsState(newMetrics)
  }, [analysisDateTime])

  const setRoutesAndSyncWarehouses = (nextRoutes: RouteDistance[]) => {
    setRoutes(nextRoutes)
    setWarehouses(prev => applyRouteTotalsToWarehouses(prev, nextRoutes))
  }

  // Load initial data from backend
  useEffect(() => {
    // Load warehouses and vehicles together so we can attach vehicleTypes to each warehouse
    Promise.all([getWarehouses(), getVehicles(), getRouteDistances()])
      .then(([warehouseList, vehicleList, routeList]) => {
        const vTypes = vehicleList.map(apiVehicleToVehicleType)
        const mappedRoutes = routeList.map(apiRouteDistanceToRouteDistance)
        const mappedWarehouses = warehouseList.map(w => ({ ...apiWarehouseToWarehouse(w), vehicles: vTypes }))
        setVehicleTypes(vTypes)
        setRoutes(mappedRoutes)
        setWarehouses(applyRouteTotalsToWarehouses(mappedWarehouses, mappedRoutes))
      })
      .catch(() => {/* backend not available, stay empty */})

    getIncomingVehicles()
      .then(res => setIncomingVehicles(res.incoming ?? []))
      .catch(() => {})

    getConfig()
      .then(cfg => {
        const catPen = (cfg as any).underload_penalty_per_unit_by_cat ?? {}
        setRiskSettingsState(prev => ({
          ...prev,
          idleCostPerMinute: typeof cfg.wait_penalty_per_minute === 'number'
            ? cfg.wait_penalty_per_minute
            : prev.idleCostPerMinute,
          emptyPenaltyPerUnit: typeof cfg.underload_penalty_per_unit === 'number'
            ? cfg.underload_penalty_per_unit
            : prev.emptyPenaltyPerUnit,
          emptyPenaltyCompact: typeof catPen.compact === 'number'
            ? catPen.compact
            : (prev.emptyPenaltyCompact ?? prev.emptyPenaltyPerUnit),
          emptyPenaltyMid: typeof catPen.mid === 'number'
            ? catPen.mid
            : (prev.emptyPenaltyMid ?? prev.emptyPenaltyPerUnit),
          emptyPenaltyLarge: typeof catPen.large === 'number'
            ? catPen.large
            : (prev.emptyPenaltyLarge ?? prev.emptyPenaltyPerUnit),
          economyThreshold: typeof cfg.economy_threshold === 'number'
            ? cfg.economy_threshold
            : prev.economyThreshold,
          confidenceLevel: typeof (cfg as any).confidence_level === 'number'
            ? (cfg as any).confidence_level
            : prev.confidenceLevel,
          granularity: typeof (cfg as any).granularity === 'number'
            ? (cfg as any).granularity
            : prev.granularity,
        }))
      })
      .catch(() => {})
  }, [])

  const setAnalysisDateTimeRounded = (value: string) => {
    setAnalysisDateTime(normalizeToHalfHour(value))
  }

  const clearCache = useCallback(() => {
    const toRemove: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k?.startsWith('dispatch_cache__') || k?.startsWith('called_')) {
        toRemove.push(k)
      }
    }
    toRemove.forEach(k => localStorage.removeItem(k))
    setWarehouseStatuses({})
  }, [])

  const refreshAbortRef = useRef<AbortController | null>(null)

  const refreshAllWarehouses = useCallback(async (
    onWarehouseComplete?: (warehouseId: string, result: ApiDispatchResponse) => void,
  ) => {
    if (refreshingAllRef.current) return
    refreshingAllRef.current = true
    setRefreshingAll(true)

    // Abort any previous refresh-all in progress
    refreshAbortRef.current?.abort()
    const controller = new AbortController()
    refreshAbortRef.current = controller

    const ts = analysisDateTime.replace('T', ' ') + ':00'
    const suffix = `__${analysisDateTime}`
    // Clear dispatch cache entries for current datetime
    const toRemove: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k?.startsWith('dispatch_cache__') && k.endsWith(suffix)) toRemove.push(k)
    }
    toRemove.forEach(k => localStorage.removeItem(k))
    setWarehouseStatuses({})
    setWarehouseMetricsState({})

    // Concurrency-limited dispatch: max 2 in parallel to avoid overloading backend
    const currentIncoming = incomingVehicles
    const queue = [...warehouses]
    const runOne = async (): Promise<void> => {
      while (queue.length > 0) {
        if (controller.signal.aborted) return
        const w = queue.shift()!
        try {
          const result = await postDispatch({
            warehouse_id: w.id,
            timestamp: ts,
            incoming_vehicles: currentIncoming.length > 0 ? currentIncoming : undefined,
            confidence_level: riskSettings.confidenceLevel,
            granularity: riskSettings.granularity,
          }, controller.signal) as ApiDispatchResponse
          if (controller.signal.aborted) return
          // Cache to localStorage
          try { localStorage.setItem('dispatch_cache__' + w.id + suffix, JSON.stringify(result)) } catch { /* quota */ }
          // Derive status
          let hasCritical = false, hasWarning = false
          for (const rp of result.routes) {
            for (const row of rp.plan) {
              if (row.demand_new > 0 && row.vehicles_count === 0) hasCritical = true
              else if (row.leftover_stock >= 1) hasWarning = true
            }
          }
          const status: 'ok' | 'warning' | 'critical' = hasCritical ? 'critical' : hasWarning ? 'warning' : 'ok'
          // Update state immediately for this warehouse
          setWarehouseStatuses(prev => ({ ...prev, [w.id]: status }))
          if (result.metrics) {
            setWarehouseMetricsState(prev => ({ ...prev, [w.id]: result.metrics! }))
          }
          // Notify caller (OptimizerPage) about this warehouse's result
          onWarehouseComplete?.(w.id, result)
        } catch (err) {
          // AbortError = user cancelled refresh; stop quietly.
          // Any other error (503 busy, network failure, etc.) = log and skip
          // this warehouse for the current refresh cycle.
          if (err instanceof Error && err.name !== 'AbortError') {
            console.warn(`[refreshAllWarehouses] warehouse ${w.id} failed:`, err.message)
          }
        }
      }
    }
    // Run 2 workers in parallel
    await Promise.all([runOne(), runOne()])
    refreshingAllRef.current = false
    setRefreshingAll(false)
  }, [analysisDateTime, incomingVehicles, warehouses, riskSettings])

  const setRiskSettings = (settings: RiskSettings) => {
    setRiskSettingsState(settings)
    patchSettings({
      wait_penalty_per_minute: settings.idleCostPerMinute,
      underload_penalty_per_unit: settings.emptyPenaltyPerUnit,
      underload_penalty_per_unit_by_cat: {
        compact: settings.emptyPenaltyCompact ?? settings.emptyPenaltyPerUnit,
        mid: settings.emptyPenaltyMid ?? settings.emptyPenaltyPerUnit,
        large: settings.emptyPenaltyLarge ?? settings.emptyPenaltyPerUnit,
      },
      economy_threshold: settings.economyThreshold,
      confidence_level: settings.confidenceLevel,
      granularity: settings.granularity,
    }).catch((err: unknown) => { console.error('[patchSettings]', err) })
  }

  const value = useMemo(
    () => ({
      warehouses,
      routes,
      setRoutes: setRoutesAndSyncWarehouses,
      vehicleTypes,
      setVehicleTypes,
      riskSettings,
      setRiskSettings,
      analysisDateTime,
      setAnalysisDateTime: setAnalysisDateTimeRounded,
      selectedWarehouseId,
      setSelectedWarehouseId,
      incomingVehicles,
      setIncomingVehicles,
      warehouseStatuses,
      setWarehouseStatus,
      warehouseMetrics,
      setWarehouseMetrics,
      clearCache,
      refreshAllWarehouses,
      refreshingAll,
    }),
    [warehouses, routes, vehicleTypes, riskSettings, analysisDateTime, selectedWarehouseId, incomingVehicles, warehouseStatuses, warehouseMetrics, clearCache, refreshAllWarehouses, refreshingAll],
  )

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  )
}

export function useSimulationContext(): SimulationContextValue {
  const ctx = useContext(SimulationContext)
  if (!ctx) {
    throw new Error('useSimulationContext must be used within SimulationProvider')
  }
  return ctx
}
