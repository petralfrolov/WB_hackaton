import { useEffect, useState } from 'react'
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
import { Trash2, Plus, Pencil, Check, X } from 'lucide-react'
import { fmt } from '../../lib/utils'
import { listVehicles, addVehicle, updateVehicle, deleteVehicle } from '../../api'

interface VehicleTypeForm {
  vehicle_type: string
  capacity_units: string
  cost_per_km: string
  underload_penalty: string
  fixed_dispatch_cost: string
}

const EMPTY_FORM: VehicleTypeForm = {
  vehicle_type: '',
  capacity_units: '',
  cost_per_km: '',
  underload_penalty: '',
  fixed_dispatch_cost: '',
}

export function VehicleTypeEditor() {
  const [vehicles, setVehicles] = useState<any[]>([])
  const [addForm, setAddForm] = useState<VehicleTypeForm>(EMPTY_FORM)
  const [editingType, setEditingType] = useState<string | null>(null)
  const [editForm, setEditForm] = useState<VehicleTypeForm>(EMPTY_FORM)
  const [showAdd, setShowAdd] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    try {
      setVehicles(await listVehicles())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  useEffect(() => { refresh() }, [])

  const handleAdd = async () => {
    const capacity = parseFloat(addForm.capacity_units)
    const cost = parseFloat(addForm.cost_per_km)
    if (!addForm.vehicle_type.trim() || isNaN(capacity) || isNaN(cost)) return
    const underload = parseFloat(addForm.underload_penalty)
    const fixed = parseFloat(addForm.fixed_dispatch_cost)
    try {
      await addVehicle({
        vehicle_type: addForm.vehicle_type.trim(),
        capacity_units: capacity,
        cost_per_km: cost,
        available: 0,
        underload_penalty: Number.isFinite(underload) ? underload : 0,
        fixed_dispatch_cost: Number.isFinite(fixed) ? fixed : 0,
      })
      setAddForm(EMPTY_FORM)
      setShowAdd(false)
      await refresh()
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
      underload_penalty: v.underload_penalty != null ? String(v.underload_penalty) : '',
      fixed_dispatch_cost: v.fixed_dispatch_cost != null ? String(v.fixed_dispatch_cost) : '',
    })
  }

  const saveEdit = async () => {
    if (!editingType) return
    const capacity = parseFloat(editForm.capacity_units)
    const cost = parseFloat(editForm.cost_per_km)
    if (isNaN(capacity) || isNaN(cost)) return
    const underload = parseFloat(editForm.underload_penalty)
    const fixed = parseFloat(editForm.fixed_dispatch_cost)
    try {
      await updateVehicle(editingType, {
        vehicle_type: editForm.vehicle_type.trim() || editingType,
        capacity_units: capacity,
        cost_per_km: cost,
        available: 0, // not managed here
        underload_penalty: Number.isFinite(underload) ? underload : undefined,
        fixed_dispatch_cost: Number.isFinite(fixed) ? fixed : undefined,
      })
      setEditingType(null)
      setEditForm(EMPTY_FORM)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const handleDelete = async (vtype: string) => {
    try {
      await deleteVehicle(vtype)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="flex flex-col gap-4 h-full overflow-y-auto">
      {error && (
        <div className="text-xs text-status-red bg-status-red/10 border border-status-red/40 rounded px-3 py-2">
          {error}
        </div>
      )}

      <div className="flex items-center justify-between gap-2">
        <div className="section-label">Типы транспортных средств</div>
        <Button size="sm" onClick={() => { setShowAdd(true); setAddForm(EMPTY_FORM) }}>
          <Plus className="w-3 h-3" /> Добавить тип
        </Button>
      </div>

      <div className="bg-surface border border-border rounded-lg overflow-hidden">
        <div className="max-h-[600px] overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Тип ТС</TableHead>
                <TableHead className="text-right">Вместимость</TableHead>
                <TableHead className="text-right">₽/км</TableHead>
                <TableHead className="text-right">Штраф недогруз ₽/ед.</TableHead>
                <TableHead className="text-right">Фикс. стоимость ₽</TableHead>
                <TableHead className="text-center w-24">—</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {showAdd && (
                <TableRow className="bg-elevated/40">
                  <TableCell>
                    <Input placeholder="Название" value={addForm.vehicle_type} onChange={e => setAddForm(f => ({ ...f, vehicle_type: e.target.value }))} />
                  </TableCell>
                  <TableCell>
                    <Input placeholder="18" value={addForm.capacity_units} onChange={e => setAddForm(f => ({ ...f, capacity_units: e.target.value }))} className="text-right" />
                  </TableCell>
                  <TableCell>
                    <Input placeholder="10" value={addForm.cost_per_km} onChange={e => setAddForm(f => ({ ...f, cost_per_km: e.target.value }))} className="text-right" />
                  </TableCell>
                  <TableCell>
                    <Input placeholder="20" value={addForm.underload_penalty} onChange={e => setAddForm(f => ({ ...f, underload_penalty: e.target.value }))} className="text-right" />
                  </TableCell>
                  <TableCell>
                    <Input placeholder="500" value={addForm.fixed_dispatch_cost} onChange={e => setAddForm(f => ({ ...f, fixed_dispatch_cost: e.target.value }))} className="text-right" />
                  </TableCell>
                  <TableCell className="text-center">
                    <div className="flex gap-1 justify-center">
                      <button onClick={handleAdd} className="p-1 rounded hover:bg-status-green/15 text-muted hover:text-status-green transition-colors"><Check className="w-4 h-4" /></button>
                      <button onClick={() => { setShowAdd(false); setAddForm(EMPTY_FORM) }} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors"><X className="w-4 h-4" /></button>
                    </div>
                  </TableCell>
                </TableRow>
              )}

              {vehicles.map(v => {
                const isEdit = editingType === v.vehicle_type
                return (
                  <TableRow key={v.vehicle_type}>
                    <TableCell className="font-semibold">
                      {isEdit ? <Input value={editForm.vehicle_type} onChange={e => setEditForm(f => ({ ...f, vehicle_type: e.target.value }))} /> : v.vehicle_type}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {isEdit ? <Input value={editForm.capacity_units} onChange={e => setEditForm(f => ({ ...f, capacity_units: e.target.value }))} className="text-right" /> : fmt(v.capacity_units)}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {isEdit ? <Input value={editForm.cost_per_km} onChange={e => setEditForm(f => ({ ...f, cost_per_km: e.target.value }))} className="text-right" /> : fmt(v.cost_per_km)}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {isEdit ? <Input value={editForm.underload_penalty} onChange={e => setEditForm(f => ({ ...f, underload_penalty: e.target.value }))} className="text-right" placeholder="—" />
                        : (v.underload_penalty != null ? fmt(v.underload_penalty) : <span className="text-muted">—</span>)}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {isEdit ? <Input value={editForm.fixed_dispatch_cost} onChange={e => setEditForm(f => ({ ...f, fixed_dispatch_cost: e.target.value }))} className="text-right" placeholder="—" />
                        : (v.fixed_dispatch_cost != null ? fmt(v.fixed_dispatch_cost) : <span className="text-muted">—</span>)}
                    </TableCell>
                    <TableCell className="text-center">
                      {isEdit ? (
                        <div className="flex gap-1 justify-center">
                          <button onClick={saveEdit} className="p-1 rounded hover:bg-status-green/15 text-muted hover:text-status-green transition-colors"><Check className="w-4 h-4" /></button>
                          <button onClick={() => { setEditingType(null); setEditForm(EMPTY_FORM) }} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors"><X className="w-4 h-4" /></button>
                        </div>
                      ) : (
                        <div className="flex gap-1 justify-center">
                          <button onClick={() => startEdit(v)} className="p-1 rounded hover:bg-elevated text-muted hover:text-foreground transition-colors"><Pencil className="w-4 h-4" /></button>
                          <button onClick={() => handleDelete(v.vehicle_type)} className="p-1 rounded hover:bg-status-red/15 text-muted hover:text-status-red transition-colors"><Trash2 className="w-4 h-4" /></button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                )
              })}

              {vehicles.length === 0 && !showAdd && (
                <TableRow>
                  <TableCell className="text-center text-muted py-6" colSpan={6}>Нет типов ТС. Добавьте первый.</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      <p className="text-xs text-muted">
        Здесь управляются характеристики типов ТС (вместимость, стоимость, штрафы).
        Количество доступных ТС настраивается для каждого склада на вкладке «Парк ТС по складам».
      </p>
    </div>
  )
}
