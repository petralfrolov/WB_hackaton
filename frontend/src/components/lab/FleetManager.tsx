import { useEffect, useMemo, useState } from 'react'
import type { Warehouse, ApiIncomingVehicle } from '../../types'
import { Input } from '../ui/input'
import { Button } from '../ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table'
import { Trash2, Plus, Clock, RefreshCw, Pencil, Check, X } from 'lucide-react'
import { cn, fmt } from '../../lib/utils'
import {
  listVehicles,
  addVehicle,
  updateVehicle,
  deleteVehicle,
  listIncoming,
  addIncoming,
  updateIncoming,
  deleteIncoming,
} from '../../lib/api'

interface FleetManagerProps {
  warehouses: Warehouse[]
}

const HORIZON_LABELS = ['Сейчас (A)', '+2ч (B)', '+4ч (C)', '+6ч (D)'] as const

interface VehicleForm {
  vehicle_type: string
  capacity_units: string
  cost_per_km: string
  available: string
  category: string
}

interface IncomingForm {
  horizon_idx: string
  vehicle_type: string
  count: string
}

const EMPTY_VEHICLE: VehicleForm = {
  vehicle_type: '',
  capacity_units: '',
  cost_per_km: '',
  available: '',
  category: 'compact',
}

const EMPTY_INCOMING: IncomingForm = {
  horizon_idx: '1',
  vehicle_type: '',
  count: '1',
}

export function FleetManager({ warehouses }: FleetManagerProps) {
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string>(warehouses[0]?.id ?? '')

  const [vehicles, setVehicles] = useState<any[]>([])
  const [incoming, setIncoming] = useState<ApiIncomingVehicle[]>([])

  const [addForm, setAddForm] = useState<VehicleForm>(EMPTY_VEHICLE)
  const [editingType, setEditingType] = useState<string | null>(null)
  const [editForm, setEditForm] = useState<VehicleForm>(EMPTY_VEHICLE)
  const [showAddVehicle, setShowAddVehicle] = useState(false)

  const [incomingForm, setIncomingForm] = useState<IncomingForm>(EMPTY_INCOMING)
  const [incomingEditingIdx, setIncomingEditingIdx] = useState<number | null>(null)
  const [showAddIncoming, setShowAddIncoming] = useState(false)

  const [error, setError] = useState<string | null>(null)

  // ---------- Load data ----------
  useEffect(() => {
    const load = async () => {
      try {
        const [vRes, iRes] = await Promise.all([listVehicles(), listIncoming()])
        setVehicles(vRes)
        setIncoming(iRes.incoming ?? [])
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      }
    }
    load()
  }, [])

  const filteredWarehouses = useMemo(() => {
    return warehouses.filter(
      w => w.name.toLowerCase().includes(search.toLowerCase()) ||
           w.city.toLowerCase().includes(search.toLowerCase()),
    )
  }, [warehouses, search])

  const refreshVehicles = async () => {
    try {
      const vRes = await listVehicles()
      setVehicles(vRes)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const refreshIncoming = async () => {
    try {
      const iRes = await listIncoming()
      setIncoming(iRes.incoming ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  // ---------- Vehicles CRUD ----------
  const handleAddVehicle = async () => {
    const capacity = parseFloat(addForm.capacity_units)
    const cost = parseFloat(addForm.cost_per_km)
    const avail = parseInt(addForm.available, 10)
    if (!addForm.vehicle_type.trim() || isNaN(capacity) || isNaN(cost) || isNaN(avail)) return
    try {
      await addVehicle({
        vehicle_type: addForm.vehicle_type.trim(),
        capacity_units: capacity,
        cost_per_km: cost,
        available: avail,
        category: addForm.category,
      })
      setAddForm(EMPTY_VEHICLE)
      setShowAddVehicle(false)
      refreshVehicles()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const startEdit = (v: any) => {
    setEditingType(v.vehicle_type)
    setEditForm({
      vehicle_type: v.vehicle_type,
      capacity_units: String(v.capacity_units),
      cost_per_km: String(v.cost_per_km),
      available: String(v.available),
      category: v.category ?? 'compact',
    })
  }

  const saveEdit = async () => {
    if (!editingType) return
    const capacity = parseFloat(editForm.capacity_units)
    const cost = parseFloat(editForm.cost_per_km)
    const avail = parseInt(editForm.available, 10)
    if (isNaN(capacity) || isNaN(cost) || isNaN(avail)) return
    try {
      await updateVehicle(editingType, {
        vehicle_type: editForm.vehicle_type.trim() || editingType,
        capacity_units: capacity,
        cost_per_km: cost,
        available: avail,
        category: editForm.category,
      })
      setEditingType(null)
      setEditForm(EMPTY_VEHICLE)
      refreshVehicles()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const deleteVehicleRow = async (vehicle_type: string) => {
    try {
      await deleteVehicle(vehicle_type)
      refreshVehicles()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  // ---------- Incoming CRUD ----------
  const handleIncomingAdd = async () => {
    const horizon = parseInt(incomingForm.horizon_idx, 10)
    const count = parseInt(incomingForm.count, 10)
    if (isNaN(horizon) || horizon < 0 || horizon > 3 || isNaN(count) || count < 1 || !incomingForm.vehicle_type.trim()) return
    try {
      await addIncoming({
        horizon_idx: horizon,
        vehicle_type: incomingForm.vehicle_type.trim(),
        count,
      })
      setIncomingForm(EMPTY_INCOMING)
      setShowAddIncoming(false)
      refreshIncoming()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const handleIncomingSave = async () => {
    if (incomingEditingIdx === null) return
    const horizon = parseInt(incomingForm.horizon_idx, 10)
    const count = parseInt(incomingForm.count, 10)
    if (isNaN(horizon) || horizon < 0 || horizon > 3 || isNaN(count) || count < 1 || !incomingForm.vehicle_type.trim()) return
    try {
      await updateIncoming(incomingEditingIdx, {
        horizon_idx: horizon,
        vehicle_type: incomingForm.vehicle_type.trim(),
        count,
      })
      setIncomingEditingIdx(null)
      setIncomingForm(EMPTY_INCOMING)
      refreshIncoming()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const handleIncomingDelete = async (idx: number) => {
    try {
      await deleteIncoming(idx)
      refreshIncoming()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const startIncomingEdit = (idx: number, item: ApiIncomingVehicle) => {
    setIncomingEditingIdx(idx)
    setIncomingForm({
      horizon_idx: String(item.horizon_idx),
      vehicle_type: item.vehicle_type,
      count: String(item.count),
    })
  }

  // ---------- Render ----------
  const vehiclesForDisplay = vehicles
  const vehicleOptions = vehiclesForDisplay.map(v => v.vehicle_type)

  return (
    <div className="flex gap-4 h-full">
      {/* Warehouse list (контекст/фильтр, парк глобальный) */}
      <div className="w-56 shrink-0 flex flex-col gap-2">
        <Input
          placeholder="Поиск склада…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div className="flex-1 overflow-y-auto space-y-0.5 rounded-lg border border-border bg-surface">
          {filteredWarehouses.map(w => (
            <button
              key={w.id}
              onClick={() => setSelectedId(w.id)}
              className={cn(
                'w-full text-left px-3 py-2.5 text-sm transition-colors',
                selectedId === w.id
                  ? 'bg-accent/10 text-accent border-l-2 border-accent'
                  : 'text-foreground hover:bg-elevated',
              )}
            >
              <div className="font-medium truncate">{w.name}</div>
              <div className="text-xs text-muted">{w.city}</div>
            </button>
          ))}
          {filteredWarehouses.length === 0 && (
            <p className="px-3 py-4 text-xs text-muted text-center">Ничего не найдено</p>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col gap-4 min-w-0 overflow-y-auto">
        {error && (
          <div className="text-xs text-status-red bg-status-red/10 border border-status-red/40 rounded px-3 py-2">
            {error}
          </div>
        )}

        <div className="flex items-center justify-between gap-2">
          <div className="section-label">
            {warehouses.find(w => w.id === selectedId)?.name ?? '—'} · Парк ТС (глобальный)
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={() => { setShowAddVehicle(true); setAddForm(EMPTY_VEHICLE) }}>
              <Plus className="w-3 h-3" /> Добавить
            </Button>
            <Button size="sm" variant="outline" onClick={() => { refreshVehicles(); refreshIncoming() }}>
              <RefreshCw className="w-3 h-3" /> Обновить
            </Button>
          </div>
        </div>

        {/* Vehicles list */}
        <div className="bg-surface border border-border rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-3 py-2 border-b border-border">
            <span className="section-label">Ожидают на складе</span>
          </div>
          <div className="max-h-96 overflow-y-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Тип ТС</TableHead>
                  <TableHead>Категория</TableHead>
                  <TableHead className="text-right">Вместимость</TableHead>
                  <TableHead className="text-right">₽/км</TableHead>
                  <TableHead className="text-right">Доступно</TableHead>
                  <TableHead className="text-center w-24">—</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {showAddVehicle && (
                  <TableRow className="bg-elevated/40">
                    <TableCell>
                      <Input
                        placeholder="gazelle_s"
                        value={addForm.vehicle_type}
                        onChange={e => setAddForm(f => ({ ...f, vehicle_type: e.target.value }))}
                      />
                    </TableCell>
                    <TableCell>
                      <select
                        value={addForm.category}
                        onChange={e => setAddForm(f => ({ ...f, category: e.target.value }))}
                        className="w-full bg-surface border border-border rounded px-2 py-1 text-sm"
                      >
                        <option value="compact">compact</option>
                        <option value="mid">mid</option>
                        <option value="large">large</option>
                      </select>
                    </TableCell>
                    <TableCell>
                      <Input
                        placeholder="18"
                        value={addForm.capacity_units}
                        onChange={e => setAddForm(f => ({ ...f, capacity_units: e.target.value }))}
                      className="text-right"
                    />
                  </TableCell>
                  <TableCell>
                    <Input
                      placeholder="40"
                      value={addForm.cost_per_km}
                      onChange={e => setAddForm(f => ({ ...f, cost_per_km: e.target.value }))}
                      className="text-right"
                    />
                  </TableCell>
                  <TableCell>
                    <Input
                      placeholder="5"
                      value={addForm.available}
                      onChange={e => setAddForm(f => ({ ...f, available: e.target.value }))}
                      className="text-right"
                    />
                  </TableCell>
                  <TableCell className="text-center">
                    <div className="flex gap-1 justify-center">
                      <button onClick={handleAddVehicle} className="p-1 rounded hover:bg-status-green/15 text-muted hover:text-status-green transition-colors" aria-label="Сохранить">
                        <Check className="w-4 h-4" />
                      </button>
                      <button onClick={() => { setShowAddVehicle(false); setAddForm(EMPTY_VEHICLE) }} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors" aria-label="Отмена">
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </TableCell>
                </TableRow>
              )}

                {vehiclesForDisplay.map(v => {
                  const isEdit = editingType === v.vehicle_type
                  return (
                    <TableRow key={v.vehicle_type}>
                      <TableCell className="font-semibold">
                      {isEdit ? (
                        <Input
                          value={editForm.vehicle_type}
                          onChange={e => setEditForm(f => ({ ...f, vehicle_type: e.target.value }))}
                        />
                      ) : v.vehicle_type}
                    </TableCell>
                    <TableCell>
                      {isEdit ? (
                        <select
                          value={editForm.category}
                          onChange={e => setEditForm(f => ({ ...f, category: e.target.value }))}
                          className="w-full bg-surface border border-border rounded px-2 py-1 text-sm"
                        >
                          <option value="compact">compact</option>
                          <option value="mid">mid</option>
                          <option value="large">large</option>
                        </select>
                      ) : (v.category ?? '—')}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {isEdit ? (
                        <Input
                          value={editForm.capacity_units}
                          onChange={e => setEditForm(f => ({ ...f, capacity_units: e.target.value }))}
                          className="text-right"
                        />
                      ) : fmt(v.capacity_units)}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {isEdit ? (
                        <Input
                          value={editForm.cost_per_km}
                          onChange={e => setEditForm(f => ({ ...f, cost_per_km: e.target.value }))}
                          className="text-right"
                        />
                      ) : fmt(v.cost_per_km)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-status-green font-semibold">
                      {isEdit ? (
                        <Input
                          value={editForm.available}
                          onChange={e => setEditForm(f => ({ ...f, available: e.target.value }))}
                          className="text-right"
                        />
                      ) : v.available}
                    </TableCell>
                    <TableCell className="text-center">
                      {isEdit ? (
                        <div className="flex gap-1 justify-center">
                          <button onClick={saveEdit} className="p-1 rounded hover:bg-status-green/15 text-muted hover:text-status-green transition-colors" aria-label="Сохранить">
                            <Check className="w-4 h-4" />
                          </button>
                          <button onClick={() => { setEditingType(null); setEditForm(EMPTY_VEHICLE) }} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors" aria-label="Отмена">
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <div className="flex gap-1 justify-center">
                          <button onClick={() => startEdit(v)} className="p-1 rounded hover:bg-elevated text-muted hover:text-foreground transition-colors" aria-label="Редактировать">
                            <Pencil className="w-4 h-4" />
                          </button>
                          <button onClick={() => deleteVehicleRow(v.vehicle_type)} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors" aria-label="Удалить">
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                )
              })}

              {vehiclesForDisplay.length === 0 && (
                <TableRow>
                  <TableCell className="text-center text-muted py-6" colSpan={6}>
                    Нет ТС. Добавьте первый тип выше.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
          </div>
        </div>

        {/* Incoming vehicles */}
        <div>
          <div className="section-label mb-1.5 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-status-yellow" />
              Прибывающие ТС (глобально)
            </span>
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={() => { setShowAddIncoming(true); setIncomingForm(EMPTY_INCOMING); setIncomingEditingIdx(null) }}>
                <Plus className="w-3 h-3" /> Добавить
              </Button>
              <Button size="sm" variant="outline" onClick={refreshIncoming}>
                <RefreshCw className="w-3 h-3" />
                Обновить
              </Button>
            </div>
          </div>

          <div className="bg-surface border border-border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Тип ТС</TableHead>
                  <TableHead>Горизонт</TableHead>
                  <TableHead className="text-right">Кол-во</TableHead>
                  <TableHead className="text-center w-16">—</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {showAddIncoming && (
                  <TableRow className="bg-elevated/40">
                    <TableCell>
                      <Input
                        list="vehicle-types"
                        placeholder="gazelle_s"
                        value={incomingForm.vehicle_type}
                        onChange={e => setIncomingForm(f => ({ ...f, vehicle_type: e.target.value }))}
                      />
                      <datalist id="vehicle-types">
                        {vehicleOptions.map(v => <option key={v} value={v} />)}
                      </datalist>
                    </TableCell>
                    <TableCell>
                      <select
                        value={incomingForm.horizon_idx}
                        onChange={e => setIncomingForm(f => ({ ...f, horizon_idx: e.target.value }))}
                        className="w-full bg-surface border border-border rounded px-2 py-1 text-sm text-foreground"
                      >
                        {HORIZON_LABELS.map((label, hi) => (
                          <option key={hi} value={hi}>{label}</option>
                        ))}
                      </select>
                    </TableCell>
                    <TableCell>
                      <Input
                        type="number"
                        min="1"
                        value={incomingForm.count}
                        onChange={e => setIncomingForm(f => ({ ...f, count: e.target.value }))}
                        className="text-right"
                      />
                    </TableCell>
                    <TableCell className="text-center">
                      <div className="flex gap-1 justify-center">
                        <button
                          onClick={incomingEditingIdx === null ? handleIncomingAdd : handleIncomingSave}
                          className="p-1 rounded hover:bg-status-green/15 text-muted hover:text-status-green transition-colors"
                          aria-label="Добавить"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => { setShowAddIncoming(false); setIncomingForm(EMPTY_INCOMING); setIncomingEditingIdx(null) }}
                          className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors"
                          aria-label="Отмена"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </TableCell>
                  </TableRow>
                )}

                {incoming.map((row, idx) => {
                  const isEdit = incomingEditingIdx === idx
                  return (
                    <TableRow key={idx}>
                      <TableCell>
                        {isEdit ? (
                          <Input
                            list="vehicle-types"
                            value={incomingForm.vehicle_type}
                            onChange={e => setIncomingForm(f => ({ ...f, vehicle_type: e.target.value }))}
                            placeholder="gazelle_s"
                          />
                        ) : row.vehicle_type}
                      </TableCell>
                      <TableCell>
                        {isEdit ? (
                          <select
                            value={incomingForm.horizon_idx}
                            onChange={e => setIncomingForm(f => ({ ...f, horizon_idx: e.target.value }))}
                            className="w-full bg-surface border border-border rounded px-2 py-1 text-sm text-foreground"
                          >
                            {HORIZON_LABELS.map((label, hi) => (
                              <option key={hi} value={hi}>{label}</option>
                            ))}
                          </select>
                        ) : HORIZON_LABELS[row.horizon_idx] ?? row.horizon_idx}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {isEdit ? (
                          <Input
                            type="number"
                            min="1"
                            value={incomingForm.count}
                            onChange={e => setIncomingForm(f => ({ ...f, count: e.target.value }))}
                            className="text-right"
                          />
                        ) : row.count}
                      </TableCell>
                      <TableCell className="text-center">
                        {isEdit ? (
                          <div className="flex gap-1 justify-center">
                            <button onClick={handleIncomingSave} className="p-1 rounded hover:bg-status-green/15 text-muted hover:text-status-green transition-colors" aria-label="Сохранить">
                              <Check className="w-4 h-4" />
                            </button>
                            <button onClick={() => { setIncomingEditingIdx(null); setIncomingForm(EMPTY_INCOMING) }} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors" aria-label="Отмена">
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ) : (
                          <div className="flex gap-1 justify-center">
                            <button onClick={() => startIncomingEdit(idx, row)} className="p-1 rounded hover:bg-elevated text-muted hover:text-foreground transition-colors" aria-label="Редактировать">
                              <Pencil className="w-4 h-4" />
                            </button>
                            <button onClick={() => handleIncomingDelete(idx)} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors" aria-label="Удалить">
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}

                {incoming.length === 0 && (
                  <TableRow>
                    <TableCell className="text-center text-muted py-4" colSpan={4}>
                      Нет прибывающих ТС.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </div>
      </div>
    </div>
  )
}
