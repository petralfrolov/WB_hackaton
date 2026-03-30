import { apiFetch } from './client'
import type {
  ApiVehicle,
  ApiIncomingVehicle,
  ApiIncomingVehicleList,
  ApiSettings,
  ApiDispatchRequest,
  ApiDispatchResponse,
  ApiWarehouseInfo,
  ApiForecastPoint,
  ApiRouteDistance,
} from '../types'

// ── Warehouses ───────────────────────────────────────────────────────────────

export function getWarehouses(): Promise<ApiWarehouseInfo[]> {
  return apiFetch<ApiWarehouseInfo[]>('/warehouses')
}

export function getWarehouse(id: string): Promise<ApiWarehouseInfo> {
  return apiFetch<ApiWarehouseInfo>(`/warehouses/${id}`)
}

export function getWarehouseForecast(id: string, timestamp: string): Promise<ApiForecastPoint[]> {
  return apiFetch<ApiForecastPoint[]>(`/warehouses/${id}/forecast?timestamp=${encodeURIComponent(timestamp)}`)
}

// ── Route distances ───────────────────────────────────────────────────────────

export function getRouteDistances(): Promise<ApiRouteDistance[]> {
  return apiFetch<ApiRouteDistance[]>('/route-distances')
}

export function putRouteDistances(items: ApiRouteDistance[]): Promise<ApiRouteDistance[]> {
  return apiFetch<ApiRouteDistance[]>('/route-distances', {
    method: 'PUT',
    body: JSON.stringify(items),
  })
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

export function postDispatch(req: ApiDispatchRequest): Promise<ApiDispatchResponse> {
  return apiFetch<ApiDispatchResponse>('/dispatch', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

// ── Vehicles ─────────────────────────────────────────────────────────────────

export function getVehicles(): Promise<ApiVehicle[]> {
  return apiFetch<ApiVehicle[]>('/vehicles')
}

export function addVehicle(v: ApiVehicle): Promise<ApiVehicle> {
  return apiFetch<ApiVehicle>('/vehicles', { method: 'POST', body: JSON.stringify(v) })
}

export function updateVehicle(type: string, v: ApiVehicle): Promise<ApiVehicle> {
  return apiFetch<ApiVehicle>(`/vehicles/${type}`, { method: 'PATCH', body: JSON.stringify(v) })
}

export function deleteVehicle(type: string): Promise<void> {
  return apiFetch<void>(`/vehicles/${type}`, { method: 'DELETE' })
}

// ── Incoming vehicles ─────────────────────────────────────────────────────────

export function getIncomingVehicles(): Promise<ApiIncomingVehicleList> {
  return apiFetch<ApiIncomingVehicleList>('/incoming-vehicles')
}

export function putIncomingVehicles(list: ApiIncomingVehicle[]): Promise<ApiIncomingVehicleList> {
  return apiFetch<ApiIncomingVehicleList>('/incoming-vehicles', {
    method: 'PUT',
    body: JSON.stringify({ incoming: list }),
  })
}

// ── Settings ──────────────────────────────────────────────────────────────────

export function getConfig(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>('/config')
}

export function patchSettings(s: Partial<ApiSettings>): Promise<void> {
  return apiFetch<void>('/settings', { method: 'PATCH', body: JSON.stringify(s) })
}
