import { useState, useCallback } from 'react'
import type { Warehouse } from '../types'
import { warehouses } from '../data/mockData'
import { WarehouseMap } from '../components/map/WarehouseMap'
import { WarehouseDrawer } from '../components/map/WarehouseDrawer'
import { Card, CardContent } from '../components/ui/card'
import { fmt } from '../lib/utils'
import { useSimulationContext } from '../context/SimulationContext'

export function MapPage() {
  const [selected, setSelected] = useState<Warehouse | null>(null)
  const { routes } = useSimulationContext()

  const handleSelect = useCallback((wh: Warehouse) => setSelected(wh), [])
  const handleClose = useCallback(() => setSelected(null), [])

  const totalWarehouses = warehouses.length
  const attentionCount = warehouses.filter(w => w.status === 'warning' || w.status === 'critical').length
  const criticalCount = warehouses.filter(w => w.status === 'critical').length

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Top KPI bar */}
      <div className="flex gap-3 px-4 py-3 border-b border-border bg-surface shrink-0">
        <KpiCard
          label="Всего складов"
          value={fmt(totalWarehouses)}
          color="#E6EDF3"
        />
        <KpiCard
          label="Требуют внимания"
          value={fmt(attentionCount)}
          color="#D29922"
        />
        <KpiCard
          label="Критических"
          value={fmt(criticalCount)}
          color="#F85149"
        />
      </div>

      {/* Map + Drawer */}
      <div className="relative flex-1 overflow-hidden">
        <WarehouseMap warehouses={warehouses} onSelect={handleSelect} />
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
