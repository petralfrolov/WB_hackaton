import { useState, useCallback } from 'react'
import type { Warehouse } from '../types'
import { WarehouseMap } from '../components/map/WarehouseMap'
import { WarehouseDrawer } from '../components/map/WarehouseDrawer'
import { Card, CardContent } from '../components/ui/card'
import { fmt } from '../lib/utils'
import { useSimulationContext } from '../context/SimulationContext'
import { RefreshCw } from 'lucide-react'
import { cn } from '../lib/utils'

export function MapPage() {
  const [selected, setSelected] = useState<Warehouse | null>(null)
  const { warehouses, routes, warehouseStatuses, refreshAllWarehouses, refreshingAll } = useSimulationContext()

  const handleSelect = useCallback((wh: Warehouse) => setSelected(wh), [])
  const handleClose = useCallback(() => setSelected(null), [])

  const totalWarehouses = warehouses.length
  const attentionCount = warehouses.filter(w => {
    const s = warehouseStatuses[w.id]
    return s === 'warning' || s === 'critical'
  }).length
  const criticalCount = warehouses.filter(w => warehouseStatuses[w.id] === 'critical').length

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Top KPI bar */}
      <div className="flex gap-3 px-4 py-3 border-b border-border bg-surface shrink-0 items-center">
        <KpiCard label="Всего складов" value={fmt(totalWarehouses)} color="#E6EDF3" />
        <KpiCard label="Прогнозируются простои товаров" value={fmt(attentionCount)} color="#D29922" />
        <KpiCard label="Прогнозируются критические простои" value={fmt(criticalCount)} color="#F85149" />
        <div className="ml-auto shrink-0">
          <button
            onClick={refreshAllWarehouses}
            disabled={refreshingAll}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium border border-border transition-colors disabled:opacity-50',
              'bg-elevated text-foreground hover:border-accent hover:text-accent',
            )}
          >
            <RefreshCw className={cn('w-3.5 h-3.5', refreshingAll && 'animate-spin')} />
            Обновить все
          </button>
        </div>
      </div>

      {/* Map + Drawer */}
      <div className="relative flex-1 overflow-hidden">
        <WarehouseMap warehouses={warehouses} onSelect={handleSelect} statusOverrides={warehouseStatuses} />
        <WarehouseDrawer warehouse={selected} onClose={handleClose} routes={routes} />
      </div>
    </div>
  )
}

function KpiCard({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color: string
}) {
  return (
    <Card className="flex-1">
      <CardContent className="pt-3">
        <div className="section-label mb-1">{label}</div>
        <span
          className="text-3xl font-bold font-mono"
          style={{ color }}
        >
          {value}
        </span>
      </CardContent>
    </Card>
  )
}
