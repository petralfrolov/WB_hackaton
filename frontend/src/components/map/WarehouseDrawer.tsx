import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { X, ArrowRight } from 'lucide-react'
import type { Warehouse, RouteDistance } from '../../types'
import { Badge } from '../ui/badge'
import type { BadgeVariant } from '../ui/badge'
import { ForecastChart } from './ForecastChart'
import { SankeyChart } from './SankeyChart'
import { fmt } from '../../lib/utils'
import { Button } from '../ui/button'
import { useSimulationContext } from '../../context/SimulationContext'

interface WarehouseDrawerProps {
  warehouse: Warehouse | null
  onClose: () => void
  routes: RouteDistance[]
}

const STATUS_LABEL: Record<Warehouse['status'], string> = {
  ok: 'В норме',
  warning: 'Предупреждение',
  critical: 'Критично',
}

const STATUS_BADGE: Record<Warehouse['status'], BadgeVariant> = {
  ok: 'ok',
  warning: 'warning',
  critical: 'critical',
}

export function WarehouseDrawer({ warehouse, onClose, routes }: WarehouseDrawerProps) {
  const drawerRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const [selectedRouteId, setSelectedRouteId] = useState('')
  const { vehicleTypes } = useSimulationContext()

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const warehouseRoutes = useMemo(
    () => (warehouse ? routes.filter(r => r.fromId === warehouse.id) : []),
    [routes, warehouse],
  )

  useEffect(() => {
    setSelectedRouteId(warehouseRoutes[0]?.id ?? '')
  }, [warehouse?.id, warehouseRoutes])

  if (!warehouse) return null

  const selectedRoute = warehouseRoutes.find(r => r.id === selectedRouteId) ?? null

  const forecast4h = warehouse.forecast.slice(0, 4).reduce((s, p) => s + p.value, 0)

  return (
    <>
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30 z-10"
        onClick={onClose}
      />

      {/* Drawer panel */}
      <div
        ref={drawerRef}
        className="absolute right-0 top-0 h-full w-[680px] bg-surface border-l border-border z-20 flex flex-col overflow-y-auto drawer-enter"
      >
        {/* Header */}
        <div className="flex items-start justify-between px-6 py-4 border-b border-border shrink-0">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-bold text-foreground">{warehouse.name}</h2>
              <Badge variant={STATUS_BADGE[warehouse.status]}>
                {STATUS_LABEL[warehouse.status]}
              </Badge>
            </div>
            <p className="text-muted text-sm mt-0.5">{warehouse.city}</p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-elevated text-muted hover:text-foreground transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="px-6 py-5 space-y-6 flex-1">
          {/* ── KPI Row ─────────────────────────────────────────────────────── */}
          <div className="grid grid-cols-2 gap-3">
            <KpiCard
              label="Ready to Ship"
              value={fmt(warehouse.readyToShip)}
              unit="ед."
              highlight
            />
            <KpiCard
              label="Прогноз на 4ч"
              value={fmt(forecast4h)}
              unit="ед."
            />
          </div>

          {/* ── Детализация button ───────────────────────────────────────────── */}
          <div className="flex justify-end">
            <Button
              size="sm"
              variant="outline"
              onClick={() => { onClose(); navigate(`/optimizer?warehouseId=${warehouse.id}`) }}
            >
              <ArrowRight className="w-3.5 h-3.5 mr-1" />
              Детализация в Оптимизаторе
            </Button>
          </div>

          {/* ── Forecast Chart ───────────────────────────────────────────────── */}
          <section>
            <div className="section-label mb-3">Прогноз отгрузок — ближайшие 8 часов</div>
            <div className="bg-elevated rounded-lg p-3">
              <ForecastChart data={warehouse.forecast} />
            </div>
          </section>

          {/* ── Sankey ──────────────────────────────────────────────────────── */}
          <section>
            <div className="section-label mb-3">Движение товаров по статусам</div>
            <div className="bg-elevated rounded-lg p-3 mb-2 border border-border/60">
              <label className="text-[11px] text-muted block mb-1">Маршрут из выбранного склада</label>
              <select
                value={selectedRouteId}
                onChange={e => setSelectedRouteId(e.target.value)}
                className="w-full h-8 rounded bg-surface border border-border px-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
              >
                {warehouseRoutes.length === 0 ? (
                  <option value="">Нет маршрутов из этого склада</option>
                ) : (
                  warehouseRoutes.map(route => (
                    <option key={route.id} value={route.id}>
                      {route.fromCity} → {route.toCity} · {route.distanceKm} км
                    </option>
                  ))
                )}
              </select>
              {selectedRoute && (
                <div className="text-[11px] text-muted mt-1.5">
                  Выбран маршрут: <span className="text-foreground">{selectedRoute.fromCity} → {selectedRoute.toCity}</span>
                </div>
              )}
            </div>
            <div className="bg-elevated rounded-lg p-4 overflow-x-auto">
              <SankeyChart data={warehouse.sankeyData} width={588} height={260} />
            </div>
            <p className="text-[11px] text-muted mt-2">
              <span style={{ color: '#D29922' }}>⬛</span> Жёлтый поток — критический отвал (&gt;30% потерь на переходе)
            </p>
          </section>

          {/* ── Vehicle Park ─────────────────────────────────────────────────── */}
          <section>
            <div className="section-label mb-3">Парк транспортных средств</div>
            <div className="bg-elevated rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-3 py-2 text-left section-label">ТС</th>
                    <th className="px-3 py-2 text-right section-label">Ёмкость</th>
                    <th className="px-3 py-2 text-right section-label">₽/км</th>
                    <th className="px-3 py-2 text-right section-label">Доступно</th>
                  </tr>
                </thead>
                <tbody>
                  {vehicleTypes.map(v => (
                    <tr key={v.id} className="border-b border-border/50 last:border-0">
                      <td className="px-3 py-2 text-foreground">{v.name}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted">{fmt(v.capacity)}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted">{fmt(v.costPerKm)}</td>
                      <td className="px-3 py-2 text-right font-mono text-accent font-semibold">{v.available}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </>
  )
}

// ── Internal KPI card ────────────────────────────────────────────────────────

function KpiCard({
  label,
  value,
  unit,
  sub,
  highlight,
}: {
  label: string
  value: string
  unit: string
  sub?: string
  highlight?: boolean
}) {
  return (
    <div className="bg-elevated rounded-lg px-4 py-3">
      <div className="section-label mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span
          className="text-3xl font-bold font-mono"
          style={{ color: highlight ? '#58A6FF' : '#E6EDF3' }}
        >
          {value}
        </span>
        <span className="text-sm text-muted">{unit}</span>
      </div>
      {sub && <div className="text-[10px] text-muted mt-0.5">{sub}</div>}
    </div>
  )
}
