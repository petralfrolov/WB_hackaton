import { apiFetch } from './client'
import type {
  ApiVehicle,
  ApiIncomingVehicle,
  ApiIncomingVehicleList,
  ApiSettings,
  ApiDispatchRequest,
  ApiDispatchResponse,
  ApiWarehouseInfo,
  ApiRouteDistance,
  ApiCallRequest,
  ApiCallResponse,
} from '../types'

// ── Warehouses ───────────────────────────────────────────────────────────────

export function getWarehouses(): Promise<ApiWarehouseInfo[]> {
  return apiFetch<ApiWarehouseInfo[]>('/warehouses')
}

export function getWarehouse(id: string): Promise<ApiWarehouseInfo> {
  return apiFetch<ApiWarehouseInfo>(`/warehouses/${id}`)
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

export function getVehicles(warehouseId?: string): Promise<ApiVehicle[]> {
  const params = warehouseId ? `?warehouse_id=${encodeURIComponent(warehouseId)}` : ''
  return apiFetch<ApiVehicle[]>(`/vehicles${params}`)
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

export function listVehicles(warehouseId?: string): Promise<ApiVehicle[]> {
  return getVehicles(warehouseId)
}

// ── Incoming vehicles ─────────────────────────────────────────────────────────

export function getIncomingVehicles(warehouseId?: string): Promise<ApiIncomingVehicleList> {
  const params = warehouseId ? `?warehouse_id=${encodeURIComponent(warehouseId)}` : ''
  return apiFetch<ApiIncomingVehicleList>(`/incoming-vehicles${params}`)
}

export function listIncoming(warehouseId?: string): Promise<ApiIncomingVehicleList> {
  return getIncomingVehicles(warehouseId)
}

export function putIncomingVehicles(list: ApiIncomingVehicle[], warehouseId?: string): Promise<ApiIncomingVehicleList> {
  const params = warehouseId ? `?warehouse_id=${encodeURIComponent(warehouseId)}` : ''
  return apiFetch<ApiIncomingVehicleList>(`/incoming-vehicles${params}`, {
    method: 'PUT',
    body: JSON.stringify({ incoming: list }),
  })
}

export async function addIncoming(item: ApiIncomingVehicle, warehouseId?: string): Promise<ApiIncomingVehicleList> {
  const current = await getIncomingVehicles(warehouseId)
  const newList = [...(current.incoming || []), item]
  return putIncomingVehicles(newList, warehouseId)
}

export async function updateIncoming(idx: number, item: ApiIncomingVehicle, warehouseId?: string): Promise<ApiIncomingVehicleList> {
  const current = await getIncomingVehicles(warehouseId)
  const newList = [...(current.incoming || [])]
  if (idx >= 0 && idx < newList.length) {
    newList[idx] = item
  }
  return putIncomingVehicles(newList, warehouseId)
}

export async function deleteIncoming(idx: number, warehouseId?: string): Promise<ApiIncomingVehicleList> {
  const current = await getIncomingVehicles(warehouseId)
  const newList = (current.incoming || []).filter((_, i) => i !== idx)
  return putIncomingVehicles(newList, warehouseId)
}

// ── Settings ──────────────────────────────────────────────────────────────────

export function getConfig(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>('/config')
}

export function patchSettings(s: Partial<ApiSettings>): Promise<void> {
  return apiFetch<void>('/settings', { method: 'PATCH', body: JSON.stringify(s) })
}

// ── Call transport (per route) ──────────────────────────────────────────────

export function callRoute(req: ApiCallRequest): Promise<ApiCallResponse> {
  return apiFetch<ApiCallResponse>('/call', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}
