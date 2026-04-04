import { useState, useCallback } from 'react'
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup,
} from 'react-simple-maps'
import type { Warehouse, ApiWarehouseMetrics } from '../../types'
import { fmt } from '../../lib/utils'
import { ZoomIn, ZoomOut, Locate } from 'lucide-react'

const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json'

const STATUS_COLORS: Record<string, string> = {
  none: '#58A6FF',
  ok: '#3FB950',
  warning: '#D29922',
  critical: '#F85149',
}

interface TooltipState {
  x: number
  y: number
  warehouse: Warehouse
}

interface WarehouseMapProps {
  warehouses: Warehouse[]
  onSelect: (warehouse: Warehouse) => void
  statusOverrides?: Record<string, 'none' | 'ok' | 'warning' | 'critical'>
  warehouseMetrics?: Record<string, ApiWarehouseMetrics>
}

export function WarehouseMap({ warehouses, onSelect, statusOverrides, warehouseMetrics }: WarehouseMapProps) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [position, setPosition] = useState<{ coordinates: [number, number]; zoom: number }>({
    coordinates: [50, 57],
    zoom: 1,
  })
  const markerScale = Math.max(0.45, Math.min(1.15, 1 / Math.sqrt(position.zoom || 1)))

  const handleMarkerEnter = useCallback(
    (e: React.MouseEvent<SVGGElement>, warehouse: Warehouse) => {
      setTooltip({ x: e.clientX + 12, y: e.clientY - 10, warehouse })
    },
    [],
  )

  const handleMarkerLeave = useCallback(() => {
    setTooltip(null)
  }, [])

  const statusLabel: Record<string, string> = {
    none: 'Прогноз не получен',
    ok: 'В норме',
    warning: 'Предупреждение',
    critical: 'Критично',
  }

  return (
    <div className="relative w-full h-full bg-surface">
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ scale: 750, center: [50, 57] }}
        style={{ width: '100%', height: '100%' }}
      >
        <ZoomableGroup
          zoom={position.zoom}
          center={position.coordinates}
          onMoveEnd={setPosition}
        >
        <Geographies geography={GEO_URL}>
          {({ geographies }) =>
            geographies
              .filter(geo => String(geo.id) === '643')
              .map(geo => (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  style={{
                    default: { fill: '#21262D', stroke: '#30363D', strokeWidth: 0.5, outline: 'none' },
                    hover: { fill: '#21262D', stroke: '#30363D', strokeWidth: 0.5, outline: 'none' },
                    pressed: { fill: '#21262D', outline: 'none' },
                  }}
                />
              ))
          }
        </Geographies>

        {warehouses.map(wh => {
          const effectiveStatus = statusOverrides?.[wh.id] ?? 'none'
          const color = STATUS_COLORS[effectiveStatus]
          return (
            <Marker
              key={wh.id}
              coordinates={[wh.lng, wh.lat]}
              onClick={() => onSelect(wh)}
              onMouseEnter={e => handleMarkerEnter(e, wh)}
              onMouseLeave={handleMarkerLeave}
              style={{ cursor: 'pointer' }}
            >
              <g>
                {/* Pulse ring */}
                {effectiveStatus === 'critical' && (
                  <circle r={9 * markerScale} fill={color} fillOpacity={0.4} className="pulse-fast" />
                )}
                {effectiveStatus === 'warning' && (
                  <circle r={9 * markerScale} fill={color} fillOpacity={0.3} className="pulse-slow" />
                )}
                {/* Core dot */}
                <circle
                  r={4 * markerScale}
                  fill={color}
                  stroke="#0D1117"
                  strokeWidth={1.5 * markerScale}
                />
              </g>
            </Marker>
          )
        })}
        </ZoomableGroup>
      </ComposableMap>

      {/* Zoom controls */}
      <div className="absolute top-3 right-3 flex flex-col gap-1 z-10">
        <button
          onClick={() => setPosition(p => ({ ...p, zoom: Math.min(p.zoom * 1.5, 10) }))}
          className="w-8 h-8 flex items-center justify-center rounded bg-elevated border border-border text-foreground hover:bg-border transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setPosition(p => ({ ...p, zoom: Math.max(p.zoom / 1.5, 0.5) }))}
          className="w-8 h-8 flex items-center justify-center rounded bg-elevated border border-border text-foreground hover:bg-border transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => setPosition({ coordinates: [50, 57], zoom: 1 })}
          className="w-8 h-8 flex items-center justify-center rounded bg-elevated border border-border text-foreground hover:bg-border transition-colors mt-1"
          title="Reset view"
        >
          <Locate className="w-4 h-4" />
        </button>
      </div>

      {/* Floating tooltip */}
      {tooltip && (
        <div
          className="pointer-events-none fixed z-50 bg-elevated border border-border rounded-lg px-3 py-2 shadow-xl text-xs"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <div className="font-semibold text-foreground text-sm">{tooltip.warehouse.name}</div>
          <div className="text-muted mt-0.5">{tooltip.warehouse.city}</div>
          <div className="mt-1.5 space-y-0.5">
            <div className="flex items-center gap-2">
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: STATUS_COLORS[statusOverrides?.[tooltip.warehouse.id] ?? 'none'] }}
              />
              <span style={{ color: STATUS_COLORS[statusOverrides?.[tooltip.warehouse.id] ?? 'none'] }}>
                {statusLabel[statusOverrides?.[tooltip.warehouse.id] ?? 'none']}
              </span>
            </div>
            <div className="text-muted">
              Готово к отгрузке:{' '}
              <span className="text-foreground font-mono font-semibold">
                {fmt(tooltip.warehouse.readyToShip)} ед.
              </span>
            </div>
            {warehouseMetrics?.[tooltip.warehouse.id] && (() => {
              const m = warehouseMetrics[tooltip.warehouse.id]
              const pct = (m.p_cover * 100).toFixed(1)
              const color = m.p_cover >= 0.9 ? '#3FB950' : m.p_cover >= 0.7 ? '#D29922' : '#F85149'
              return (
                <div className="mt-1 flex items-center gap-1.5">
                  <span className="text-muted text-xs">P(хватит транспорта):</span>
                  <span className="font-mono font-semibold text-xs" style={{ color }}>{pct}%</span>
                </div>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}
