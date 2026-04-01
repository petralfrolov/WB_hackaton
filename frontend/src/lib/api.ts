const API_BASE = (import.meta as any)?.env?.VITE_API_URL ?? 'http://localhost:8000'

async function handleJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export async function listVehicles() {
  const res = await fetch(`${API_BASE}/vehicles`)
  return handleJson<any>(res)
}

export async function addVehicle(v: any) {
  const res = await fetch(`${API_BASE}/vehicles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(v),
  })
  return handleJson<any>(res)
}

export async function updateVehicle(type: string, v: any) {
  const res = await fetch(`${API_BASE}/vehicles/${encodeURIComponent(type)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(v),
  })
  return handleJson<any>(res)
}

export async function deleteVehicle(type: string) {
  const res = await fetch(`${API_BASE}/vehicles/${encodeURIComponent(type)}`, {
    method: 'DELETE',
  })
  return handleJson<any>(res)
}

export async function listIncoming() {
  const res = await fetch(`${API_BASE}/incoming-vehicles`)
  return handleJson<any>(res)
}

export async function addIncoming(v: any) {
  const res = await fetch(`${API_BASE}/incoming-vehicles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(v),
  })
  return handleJson<any>(res)
}

export async function updateIncoming(idx: number, v: any) {
  const res = await fetch(`${API_BASE}/incoming-vehicles/${idx}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(v),
  })
  return handleJson<any>(res)
}

export async function deleteIncoming(idx: number) {
  const res = await fetch(`${API_BASE}/incoming-vehicles/${idx}`, {
    method: 'DELETE',
  })
  return handleJson<any>(res)
}

export async function callRoute(payload: { route_id: string; timestamp: string; warehouse_id?: string }) {
  const res = await fetch(`${API_BASE}/call`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleJson<any>(res)
}
