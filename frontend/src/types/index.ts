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
  category?: 'small' | 'medium' | 'large'
  underloadPenalty?: number
  fixedDispatchCost?: number
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
  readyToShip: number
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

export type Granularity = 0.5 | 1 | 2

export interface RiskSettings {
  economyThreshold: number
  maxWaitMinutes: number
  idleCostPerMinute: number
  emptyPenaltyPerUnit: number
  confidenceLevel: number // 0.9 = 90% доверительная вероятность
  emptyPenaltyCompact?: number
  emptyPenaltyMid?: number
  emptyPenaltyLarge?: number
  granularity: Granularity
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
  ready_to_ship: number
}

export interface ApiVehicle {
  vehicle_type: string
  capacity_units: number
  cost_per_km: number
  available: number
  category?: 'small' | 'medium' | 'large'
  underload_penalty?: number
  fixed_dispatch_cost?: number
}

export interface ApiIncomingVehicle {
  horizon_idx: number
  vehicle_type: string
  count: number
  original_horizon_idx?: number  // original index before granularity remap
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
  confidence_level?: number
  granularity?: number
  underload_penalty_per_unit_by_cat?: {
    compact?: number
    mid?: number
    large?: number
  }
}

export interface ApiPlanRow {
  office_from_id: string | null
  route_id: string
  timestamp: string
  horizon: string
  vehicle_type: string
  vehicles_count: number
  demand_new: number
  demand_lower: number
  demand_upper: number
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
  confidence_level?: number
  granularity?: number
}

export interface ApiRouteMetrics {
  route_id: string
  fill_rate: number   // 0–1
  cpo: number         // ₽ per shipped unit
}

export interface ApiWarehouseMetrics {
  p_cover: number                  // 0–1, probability capacity suffices across all horizons
  p_cover_by_horizon: number[]     // dynamic length based on granularity
  fill_rate: number                // 0–1
  cpo: number                      // ₽ per shipped unit
  route_metrics: ApiRouteMetrics[]
  horizon_labels?: string[]        // dynamic labels from backend
  fleet_utilization_ratio?: number   // required_capacity / available_capacity
  fleet_capacity_shortfall?: number  // required_capacity - available_capacity (units)
  required_capacity_units?: number   // total dispatched vehicle capacity (units)
  available_capacity_units?: number  // total available fleet capacity (units)
}

export interface ApiDispatchResponse {
  warehouse_id: string
  office_from_id: string
  timestamp: string
  routes: ApiRoutePlan[]
  total_cost: number
  metrics?: ApiWarehouseMetrics
  granularity?: number
  horizon_labels?: string[]
}

export interface ApiCallRequest {
  route_id: string
  timestamp: string
  warehouse_id?: string
}

export interface ApiCallVehicle {
  vehicle_type: string
  vehicles_count: number
  category?: string
  capacity_units: number
  cost_per_km: number
  empty_capacity_units: number
  cost_fixed: number
  cost_underload: number
}

export interface ApiCallPayload {
  route_id: string
  office_from_id?: string
  dispatch_time: string
  horizon: string
  vehicles: ApiCallVehicle[]
  costs: { fixed: number; underload: number; wait: number; total: number }
  demand: { ready_to_ship: number; pred_0_2h: number; pred_2_4h: number; pred_4_6h: number }
}

export interface ApiCallResponse {
  request: ApiCallPayload
}
