import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
import type {
  CostBreakdown,
  CostBreakdownVehicle,
  CostScenario,
  Granularity,
  RiskSettings,
  ApiWarehouseInfo,
  ApiVehicle,
  ApiDispatchResponse,
  ApiPlanRow,
  Warehouse,
  VehicleType,
  RouteDistance,
  ApiRouteDistance,
} from '../types'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function fmt(n: number, opts?: Intl.NumberFormatOptions) {
  return new Intl.NumberFormat('ru-RU', opts).format(n)
}

export function fmtCurrency(n: number) {
  return new Intl.NumberFormat('ru-RU', {
    style: 'currency',
    currency: 'RUB',
    maximumFractionDigits: 0,
  }).format(n)
}

// ─── Granularity / horizon helpers ────────────────────────────────────────────

/** Build the horizon labels that the backend generates for a given granularity. */
export function makeHorizonLabels(granularity: Granularity): string[] {
  if (granularity === 2) return ['A: now', 'B: +2h', 'C: +4h', 'D: +6h']
  const abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  const period = granularity
  const nFuture = Math.round(6 / period)
  const labels = ['A: now']
  for (let i = 1; i <= nFuture; i++) {
    const h = i * period
    const hStr = h === Math.floor(h) ? `${h}h` : `${h}h`
    labels.push(`${abc[i]}: +${hStr}`)
  }
  return labels
}

/** Human-readable display labels for horizon labels. */
export function horizonDisplayLabel(label: string): string {
  if (label === 'A: now') return 'Сейчас'
  // "B: +2h" → "+2ч", "C: +0.5h" → "+0.5ч"
  const m = label.match(/\+(.+)h$/)
  return m ? `+${m[1]}ч` : label
}

/** Get the future horizon keys (excluding "A: now") from dispatch result or granularity. */
export function getFutureHorizonKeys(dispatchResult?: ApiDispatchResponse | null, granularity: Granularity = 2): string[] {
  if (dispatchResult?.horizon_labels) {
    return dispatchResult.horizon_labels.filter((l: string) => l !== 'A: now')
  }
  return makeHorizonLabels(granularity).filter(l => l !== 'A: now')
}

/** Map original 4-horizon incoming vehicle index to new horizon count. */
export function mapIncomingHorizonIdx(
  originalIdx: number,
  fromGranularity: Granularity,
  toGranularity: Granularity,
): number {
  // Convert original index to hours, then to new index
  const hours = originalIdx * fromGranularity
  return Math.round(hours / toGranularity)
}

// ─── Default risk settings ────────────────────────────────────────────────────

export const defaultRiskSettings: RiskSettings = {
  economyThreshold: 0,
  maxWaitMinutes: 55,
  idleCostPerMinute: 8,
  emptyPenaltyPerUnit: 12,
  confidenceLevel: 0.9, // 90% по умолчанию
  routeCorrelation: 0.3,
  granularity: 2,
}

// ─── Cost helpers ─────────────────────────────────────────────────────────────

export function computeCostFromBreakdown(b: CostBreakdown): number {
  const fixed = b.vehicles.reduce((s, v) => s + v.fixedCost * v.count, 0)
  const empty = b.wEcon * b.vehicles.reduce((s, v) => s + (v.capacity - v.load) * v.count, 0) * b.pEmpty
  const delay = b.wUrg * b.itemsWaiting * b.avgWaitMinutes * b.pDelay
  return Math.round(fixed + empty + delay)
}

function makeBreakdown(
  vehicles: Array<{ name: string; count: number; fixedCost: number; capacity: number; load: number }>,
  wEcon = 1.0,
  wUrg = 1.2,
  avgWait = 45,
  items = 120,
): CostBreakdown {
  return { vehicles, wEcon, wUrg, pEmpty: 12, pDelay: 8, avgWaitMinutes: avgWait, itemsWaiting: items }
}

const FIXED_SCENARIO_META = [
  { id: 'a', name: 'Вариант А', waitMinutes: 0,   time: 'Сейчас', description: 'Вызвать машины сейчас' },
  { id: 'b', name: 'Вариант Б', waitMinutes: 120, time: '+2 ч',   description: 'Вызвать через 2 часа' },
  { id: 'c', name: 'Вариант В', waitMinutes: 240, time: '+4 ч',   description: 'Вызвать через 4 часа' },
  { id: 'd', name: 'Вариант Г', waitMinutes: 360, time: '+6 ч',   description: 'Вызвать через 6 часов' },
] as const

function vehicleMixLabel(vehicles: CostBreakdown['vehicles']): string {
  return vehicles.map(v => `${v.count}× ${v.name}`).join(' + ')
}

function defaultBaseScenario(forecast: number): CostScenario {
  return {
    id: 'a', name: 'Вариант А', description: 'Базовый микс', cost: 11200, time: 'Сейчас',
    breakdown: makeBreakdown([
      { name: 'Газель', count: 2, fixedCost: 2800, capacity: 50, load: 45 },
      { name: 'Ларгус', count: 1, fixedCost: 1800, capacity: 20, load: 16 },
    ], 0.8, 1.2, 0, forecast),
  }
}

export function getCostScenarios(
  forecast = 300,
  options?: { riskSettings?: RiskSettings; itemsReadyNow?: number },
): CostScenario[] {
  const base = defaultBaseScenario(forecast)
  const risk = options?.riskSettings ?? defaultRiskSettings
  const waitingItems = options?.itemsReadyNow ?? base.breakdown.itemsWaiting ?? forecast
  const baseVehicles = base.breakdown.vehicles

  return FIXED_SCENARIO_META.map((meta, idx) => {
    const vehicles = baseVehicles.map(v => {
      const headroom = Math.max(0, v.capacity - v.load)
      const loaded = Math.min(v.capacity, Math.round(v.load + headroom * Math.min(idx * 0.35, 1)))
      return { ...v, load: loaded }
    })
    const breakdown: CostBreakdown = {
      ...base.breakdown,
      vehicles,
      wEcon: Math.max(0.2, risk.economyThreshold / 100),
      pEmpty: risk.emptyPenaltyPerUnit,
      pDelay: risk.idleCostPerMinute,
      avgWaitMinutes: meta.waitMinutes,
      itemsWaiting: waitingItems,
    }
    return {
      id: meta.id,
      name: meta.name,
      description: `${vehicleMixLabel(vehicles)} · ${meta.description}`,
      cost: computeCostFromBreakdown(breakdown),
      time: meta.time,
      breakdown,
    }
  })
}

// ─── API → UI converters ──────────────────────────────────────────────────────

export function apiWarehouseToWarehouse(w: ApiWarehouseInfo): Warehouse {
  return {
    id: w.id,
    name: w.name,
    city: w.city,
    lat: w.lat ?? 0,
    lng: w.lng ?? 0,
    status: 'ok',  // status is now derived from dispatch results, not from API
    readyToShip: w.ready_to_ship,
    forecast: [],
    vehicles: [],
  }
}

export function apiRouteDistanceToRouteDistance(r: ApiRouteDistance): RouteDistance {
  return {
    id: r.id,
    fromId: r.from_id,
    toId: r.to_id,
    fromCity: r.from_city,
    toCity: r.to_city,
    distanceKm: r.distance_km,
    readyToShip: r.ready_to_ship,
  }
}

export function routeDistanceToApi(r: RouteDistance): ApiRouteDistance {
  return {
    id: r.id,
    from_id: r.fromId,
    to_id: r.toId,
    from_city: r.fromCity,
    to_city: r.toCity,
    distance_km: r.distanceKm,
    ready_to_ship: r.readyToShip,
  }
}

const VEHICLE_DISPLAY_NAMES: Record<string, string> = {
  gazelle_s: 'Газель S',
  gazelle_l: 'Газель L',
  truck_m: 'Грузовик M',
  truck_l: 'Грузовик L',
}

const VEHICLE_CAPACITY_UNITS: Record<string, number> = {
  gazelle_s: 18,
  gazelle_l: 26,
  truck_m: 40,
  truck_l: 60,
}

export function apiVehicleToVehicleType(v: ApiVehicle): VehicleType {
  return {
    id: v.vehicle_type,
    name: VEHICLE_DISPLAY_NAMES[v.vehicle_type] ?? v.vehicle_type,
    capacity: v.capacity_units,
    costPerKm: v.cost_per_km,
    available: v.available,
    underloadPenalty: v.underload_penalty,
    fixedDispatchCost: v.fixed_dispatch_cost,
  }
}

const DISPATCH_HORIZON_META: Record<string, { name: string; time: string; description: string }> = {
  'A: now': { name: 'Сейчас',  time: 'Сейчас', description: 'Отправить немедленно' },
  'B: +2h': { name: '+2 ч',   time: '+2 ч',   description: 'Отправить через 2 часа' },
  'C: +4h': { name: '+4 ч',   time: '+4 ч',   description: 'Отправить через 4 часа' },
  'D: +6h': { name: '+6 ч',   time: '+6 ч',   description: 'Отправить через 6 часов' },
}

const HORIZON_ORDER = ['A: now', 'B: +2h', 'C: +4h', 'D: +6h']

export function dispatchToScenarios(
  result: ApiDispatchResponse,
  riskSettings: RiskSettings,
): CostScenario[] {
  // Group all plan rows across all routes by horizon
  const byHorizon = new Map<string, ApiPlanRow[]>()
  for (const route of result.routes) {
    for (const row of route.plan) {
      let rows = byHorizon.get(row.horizon)
      if (!rows) { rows = []; byHorizon.set(row.horizon, rows) }
      rows.push(row)
    }
  }

  const wUrg = 1.0
  const wEcon = Math.max(0.2, riskSettings.economyThreshold / 100)
  const pEmpty = riskSettings.emptyPenaltyPerUnit
  const pDelay = riskSettings.idleCostPerMinute

  return HORIZON_ORDER.filter(h => byHorizon.has(h)).map(h => {
    const rows = byHorizon.get(h)!
    const meta = DISPATCH_HORIZON_META[h] ?? { name: h, time: h, description: h }

    const totalCost = rows.reduce((s, r) => s + r.cost_total, 0)
    const totalWait = rows.reduce((s, r) => s + r.cost_wait, 0)
    const totalDemand = rows.reduce((s, r) => s + r.demand_new, 0)
    const totalShipped = rows.reduce((s, r) => s + r.actually_shipped, 0)
    const itemsWaiting = Math.max(0, Math.round(totalDemand - totalShipped))

    // Derive avgWaitMinutes from actual wait cost, fall back to 0 if nothing waiting
    const avgWaitMinutes = itemsWaiting > 0 && pDelay > 0
      ? Math.round(totalWait / (wUrg * itemsWaiting * pDelay))
      : 0

    // Aggregate rows by vehicle type
    const vehicleMap = new Map<string, { count: number; costFixed: number; emptyCapacity: number }>()
    for (const row of rows) {
      if (row.vehicle_type === 'none' || row.vehicles_count <= 0) continue
      const existing = vehicleMap.get(row.vehicle_type)
      if (existing) {
        existing.count += row.vehicles_count
        existing.costFixed += row.cost_fixed
        existing.emptyCapacity += row.empty_capacity_units
      } else {
        vehicleMap.set(row.vehicle_type, {
          count: row.vehicles_count,
          costFixed: row.cost_fixed,
          emptyCapacity: row.empty_capacity_units,
        })
      }
    }

    const vehicles: CostBreakdownVehicle[] = Array.from(vehicleMap.entries()).map(([vtype, data]) => {
      const capacity = VEHICLE_CAPACITY_UNITS[vtype] ?? 30
      const emptyPerVehicle = data.count > 0 ? data.emptyCapacity / data.count : 0
      const load = Math.max(0, Math.round(capacity - emptyPerVehicle))
      return {
        name: VEHICLE_DISPLAY_NAMES[vtype] ?? vtype,
        count: data.count,
        fixedCost: data.count > 0 ? Math.round(data.costFixed / data.count) : 0,
        capacity,
        load,
      }
    })

    const breakdown: CostBreakdown = {
      vehicles,
      wEcon,
      wUrg,
      pEmpty,
      pDelay,
      avgWaitMinutes,
      itemsWaiting,
    }

    return {
      id: `dispatch-${h}`,
      name: meta.name,
      description: meta.description,
      cost: Math.round(totalCost),
      time: meta.time,
      breakdown,
    }
  })
}
