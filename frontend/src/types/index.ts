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
