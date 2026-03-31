import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { ApiIncomingVehicle, RouteDistance, RiskSettings, VehicleType, Warehouse } from '../types'
import { getWarehouses, getRouteDistances, getConfig, getVehicles, getIncomingVehicles, patchSettings } from '../api'
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
  const [analysisDateTime, setAnalysisDateTime] = useState<string>('2025-03-13T11:00')
  const [selectedWarehouseId, setSelectedWarehouseId] = useState<string | null>(null)
  const [incomingVehicles, setIncomingVehicles] = useState<ApiIncomingVehicle[]>([])
  const [warehouseStatuses, setWarehouseStatuses] = useState<Record<string, 'none' | 'ok' | 'warning' | 'critical'>>({});

  const setWarehouseStatus = useCallback((id: string, status: 'none' | 'ok' | 'warning' | 'critical') => {
    setWarehouseStatuses(prev => prev[id] === status ? prev : { ...prev, [id]: status })
  }, [])

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
          maxWaitMinutes: typeof cfg.max_wait_minutes === 'number'
            ? cfg.max_wait_minutes
            : prev.maxWaitMinutes,
        }))
      })
      .catch(() => {})
  }, [])

  const setAnalysisDateTimeRounded = (value: string) => {
    setAnalysisDateTime(normalizeToHalfHour(value))
  }

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
      max_wait_minutes: settings.maxWaitMinutes,
    }).catch(() => {})
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
    }),
    [warehouses, routes, vehicleTypes, riskSettings, analysisDateTime, selectedWarehouseId, incomingVehicles, warehouseStatuses],
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
