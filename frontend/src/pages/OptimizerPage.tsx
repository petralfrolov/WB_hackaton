import { useState, useCallback, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { CostScenario, ApiDispatchResponse } from '../types'
import { dispatchToScenarios, cn } from '../lib/utils'
import { RouteTable } from '../components/optimizer/RouteTable'
import { CostBenefitCard } from '../components/optimizer/CostBenefitCard'
import { useSimulationContext } from '../context/SimulationContext'
import { postDispatch } from '../api'

export function OptimizerPage() {
  const [searchParams] = useSearchParams()
  const { warehouses, routes, riskSettings, analysisDateTime, setSelectedWarehouseId, incomingVehicles } = useSimulationContext()

  const [warehouseId, setWarehouseId] = useState<string>(searchParams.get('warehouseId') ?? '')
  const [dispatchResult, setDispatchResult] = useState<ApiDispatchResponse | null>(null)
  const [dispatchLoading, setDispatchLoading] = useState(false)
  const [dispatchError, setDispatchError] = useState<string | null>(null)
  const [scenarios, setScenarios] = useState<CostScenario[]>([])
  const [search, setSearch] = useState('')

  const selectedWarehouse = warehouses.find(w => w.id === warehouseId) ?? null
  const warehouseRoutes = selectedWarehouse ? routes.filter(r => r.fromId === selectedWarehouse.id) : []

  // ── Dispatch logic ───────────────────────────────────────────────────────
  const runDispatch = useCallback(async (wid: string) => {
    if (!wid) return
    setDispatchLoading(true)
    setDispatchError(null)
    try {
      const ts = analysisDateTime.replace('T', ' ') + ':00'
      const result = await postDispatch({
        warehouse_id: wid,
        timestamp: ts,
        incoming_vehicles: incomingVehicles.length > 0 ? incomingVehicles : undefined,
      })
      setDispatchResult(result)
      setScenarios(dispatchToScenarios(result, riskSettings))
    } catch (err) {
      setDispatchError(err instanceof Error ? err.message : String(err))
      setDispatchResult(null)
    } finally {
      setDispatchLoading(false)
    }
  }, [analysisDateTime, incomingVehicles, riskSettings])

  const runDispatchRef = useRef(runDispatch)
  useEffect(() => { runDispatchRef.current = runDispatch }, [runDispatch])

  // Auto-dispatch when warehouse is selected
  const handleSelectWarehouse = useCallback((id: string) => {
    setWarehouseId(id)
    setSelectedWarehouseId(id)
    setDispatchResult(null)
    setScenarios([])
    runDispatchRef.current(id)
  }, [setSelectedWarehouseId])

  // Re-dispatch on datetime change if warehouse already chosen
  const isFirstRender = useRef(true)
  useEffect(() => {
    if (isFirstRender.current) { isFirstRender.current = false; return }
    if (warehouseId) runDispatchRef.current(warehouseId)
  }, [analysisDateTime]) // eslint-disable-line react-hooks/exhaustive-deps

  // Sync URL param → auto-dispatch when navigated from map drawer
  useEffect(() => {
    const id = searchParams.get('warehouseId') ?? ''
    if (id && id !== warehouseId) {
      handleSelectWarehouse(id)
    }
  }, [searchParams]) // eslint-disable-line react-hooks/exhaustive-deps

  const STATUS_COLOR: Record<string, string> = {
    ok: 'bg-status-green',
    warning: 'bg-status-yellow',
    critical: 'bg-status-red',
  }

  const filteredWarehouses = warehouses.filter(w =>
    w.name.toLowerCase().includes(search.toLowerCase()) ||
    w.city.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-surface shrink-0">
        <h1 className="text-base font-semibold text-foreground">Оптимизатор транспортных вызовов</h1>
        <p className="text-xs text-muted mt-0.5">
          Выберите склад — прогноз и план развозки загрузятся автоматически.
        </p>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden divide-x divide-border">
        {/* ── Warehouse sidebar ──────────────────────────────────────────── */}
        <div className="w-52 shrink-0 flex flex-col overflow-hidden bg-surface">
          <div className="px-3 py-2 border-b border-border">
            <input
              type="text"
              placeholder="Поиск склада…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full bg-elevated border border-border rounded px-2 py-1.5 text-xs text-foreground focus:outline-none focus:border-accent"
            />
          </div>
          <div className="flex-1 overflow-y-auto">
            {filteredWarehouses.map(w => (
              <button
                key={w.id}
                onClick={() => handleSelectWarehouse(w.id)}
                className={cn(
                  'w-full text-left px-3 py-2.5 transition-colors border-l-2',
                  warehouseId === w.id
                    ? 'bg-accent/10 border-accent'
                    : 'hover:bg-elevated border-transparent',
                )}
              >
                <div className="flex items-center gap-2">
                  <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', STATUS_COLOR[w.status])} />
                  <span className={cn('font-medium truncate text-xs', warehouseId === w.id ? 'text-accent' : 'text-foreground')}>
                    {w.name}
                  </span>
                </div>
                <div className="text-[11px] text-muted ml-3.5">{w.city}</div>
              </button>
            ))}
          </div>
        </div>

        {/* ── Main content ───────────────────────────────────────────────── */}
        <div className="flex-1 flex overflow-hidden divide-x divide-border">
          {/* Route table */}
          <div className="flex-[2] min-w-0 overflow-y-auto p-4">
            {!warehouseId ? (
              <div className="flex items-center justify-center h-full text-muted text-sm">
                Выберите склад слева для просмотра маршрутов и прогноза
              </div>
            ) : (
              <RouteTable
                warehouse={selectedWarehouse}
                warehouseRoutes={warehouseRoutes}
                dispatchResult={dispatchResult}
                loading={dispatchLoading}
                error={dispatchError}
              />
            )}
          </div>

          {/* Cost scenarios */}
          <div className="flex-1 min-w-[280px] overflow-y-auto p-4">
            <CostBenefitCard recommendation={null} scenarios={scenarios} />
          </div>
        </div>
      </div>
    </div>
  )
}
