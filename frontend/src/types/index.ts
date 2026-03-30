export interface ForecastPoint {
  time: string
  value: number
  lower: number
  upper: number
}

export interface SankeyNodeDatum {
  id: string
  label: string
}

export interface SankeyLinkDatum {
  source: string
  target: string
  value: number
}

export interface SankeyData {
  nodes: SankeyNodeDatum[]
  links: SankeyLinkDatum[]
}

export interface IncomingVehicle {
  id: string
  name: string
  arrivalMinutes: number
  count: number
}

export interface VehicleType {
  id: string
  name: string
  capacity: number
  costPerKm: number
  available: number
  incoming?: IncomingVehicle[]
}

export interface Warehouse {
  id: string
  name: string
  city: string
  lat: number
  lng: number
  status: 'ok' | 'warning' | 'critical'
  readyToShip: number
  forecast: ForecastPoint[]
  sankeyData: SankeyData
  vehicles: VehicleType[]
}

export interface TransportRecommendation {
  id: string
  route: string
  warehouseId: string
  forecast: number
  recommendation: string
  status: 'pending' | 'called' | 'warning'
}

export interface RouteDistance {
  id: string
  fromId: string
  toId: string
  fromCity: string
  toCity: string
  distanceKm: number
}

export interface CostBreakdownVehicle {
  name: string
  count: number
  fixedCost: number
  capacity: number
  load: number
}

export interface CostBreakdown {
  vehicles: CostBreakdownVehicle[]
  wEcon: number
  wUrg: number
  pEmpty: number
  pDelay: number
  avgWaitMinutes: number
  itemsWaiting: number
}

export interface CostScenario {
  id: string
  name: string
  description: string
  cost: number
  time: string
  breakdown: CostBreakdown
}

export interface RiskSettings {
  economyThreshold: number
  maxWaitMinutes: number
  idleCostPerMinute: number
  emptyPenaltyPerUnit: number
}

// ── Backend API types (mirrors Pydantic schemas) ─────────────────────────────

export interface ApiWarehouseInfo {
  id: string
  name: string
  city: string
  lat: number
  lng: number
  office_from_id: string
  route_ids: string[]
  status: 'ok' | 'warning' | 'critical'
  ready_to_ship: number
}

export interface ApiForecastPoint {
  time: string
  value: number
  lower: number
  upper: number
}

export interface ApiRouteDistance {
  id: string
  from_id: string
  to_id: string
  from_city: string
  to_city: string
  distance_km: number
}

export interface ApiVehicle {
  vehicle_type: string
  capacity_units: number
  cost_per_km: number
  available: number
}

export interface ApiIncomingVehicle {
  horizon_idx: 0 | 1 | 2 | 3
  vehicle_type: string
  count: number
}

export interface ApiIncomingVehicleList {
  incoming: ApiIncomingVehicle[]
}

export interface ApiSettings {
  underload_penalty_per_unit: number
  wait_penalty_per_minute: number
  initial_stock_units: number
  route_distance_km: number
  economy_threshold?: number
  max_wait_minutes?: number
}

export interface ApiPlanRow {
  office_from_id: string | null
  route_id: string
  timestamp: string
  horizon: string
  vehicle_type: string
  vehicles_count: number
  demand_new: number
  demand_carried_over: number
  total_available: number
  actually_shipped: number
  leftover_stock: number
  empty_capacity_units: number
  cost_fixed: number
  cost_underload: number
  cost_wait: number
  cost_total: number
}

export interface ApiRoutePlan {
  route_id: string
  plan: ApiPlanRow[]
  coverage_min: number
}

export interface ApiDispatchRequest {
  warehouse_id: string
  timestamp: string
  vehicles_override?: ApiVehicle[]
  incoming_vehicles?: ApiIncomingVehicle[]
  wait_penalty_per_minute?: number
  underload_penalty_per_unit?: number
  global_fleet?: boolean
}

export interface ApiDispatchResponse {
  warehouse_id: string
  office_from_id: string
  timestamp: string
  routes: ApiRoutePlan[]
  total_cost: number
}
