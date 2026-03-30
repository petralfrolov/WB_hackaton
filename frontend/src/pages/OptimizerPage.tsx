import { useState, useCallback, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { TransportRecommendation, CostScenario, ApiDispatchResponse } from '../types'
import { dispatchToScenarios } from '../lib/utils'
import { RecommendationTable } from '../components/optimizer/RecommendationTable'
import { CostBenefitCard } from '../components/optimizer/CostBenefitCard'
import { useSimulationContext } from '../context/SimulationContext'
import { postDispatch } from '../api'

const HORIZON_LABELS: Record<string, string> = {
  'A: now': 'Сейчас',
  'B: +2h': '+2ч',
  'C: +4h': '+4ч',
  'D: +6h': '+6ч',
}

export function OptimizerPage() {
  const [searchParams] = useSearchParams()
  const { warehouses, riskSettings, analysisDateTime, selectedWarehouseId, setSelectedWarehouseId, incomingVehicles } = useSimulationContext()
  const recs: TransportRecommendation[] = warehouses.map(w => ({
    id: w.id,
    route: `${w.city} → Маршрут`,
    warehouseId: w.id,
    forecast: w.readyToShip,
    recommendation: 'Нажмите «Получить план»',
    status: w.status === 'critical' ? 'warning' as const : 'pending' as const,
  }))
  const [selectedRec, setSelectedRec] = useState<TransportRecommendation | null>(null)
  const [scenarios, setScenarios] = useState<CostScenario[]>([])
  const [warehouseFilter, setWarehouseFilter] = useState<string>(searchParams.get('warehouseId') ?? '')

  const [dispatchResult, setDispatchResult] = useState<ApiDispatchResponse | null>(null)
  const [dispatchLoading, setDispatchLoading] = useState(false)
  const [dispatchError, setDispatchError] = useState<string | null>(null)

  // Active warehouse: prefer URL param, fall back to context
  const activeWarehouseId = warehouseFilter || selectedWarehouseId || ''

  // Sync URL param when it changes (e.g. navigation from drawer)
  useEffect(() => {
    const id = searchParams.get('warehouseId') ?? ''
    setWarehouseFilter(id)
    if (id) setSelectedWarehouseId(id)
  }, [searchParams])

  useEffect(() => {
    if (!selectedRec) return
    // Only fall back to mock scenarios if no dispatch result is available yet
    if (!dispatchResult) return
    setScenarios(dispatchToScenarios(dispatchResult, riskSettings))
  }, [riskSettings, selectedRec, dispatchResult])

  const warehouseName = activeWarehouseId
    ? warehouses.find(w => w.id === activeWarehouseId)?.name ?? activeWarehouseId
    : null
  const warehouseLabelById = Object.fromEntries(warehouses.map(w => [w.id, w.name]))

  const handleSelect = useCallback((rec: TransportRecommendation) => {
    setSelectedRec(rec)
    // Also set active warehouse so the dispatch button becomes enabled
    setWarehouseFilter(rec.warehouseId)
    setSelectedWarehouseId(rec.warehouseId)
  }, [setSelectedWarehouseId])

  const [calledIds, setCalledIds] = useState<Set<string>>(new Set())
  const handleCall = useCallback((id: string) => {
    setCalledIds(prev => new Set([...prev, id]))
    setSelectedRec(prev => (prev?.id === id ? { ...prev, status: 'called' as const } : prev))
  }, [])

  const displayRecs = recs.map(r => calledIds.has(r.id) ? { ...r, status: 'called' as const } : r)

  const handleDispatch = useCallback(async () => {
    if (!activeWarehouseId || !analysisDateTime) return
    setDispatchLoading(true)
    setDispatchError(null)
    try {
      const ts = analysisDateTime.replace('T', ' ') + ':00'
      const result = await postDispatch({
        warehouse_id: activeWarehouseId,
        timestamp: ts,
        incoming_vehicles: incomingVehicles.length > 0 ? incomingVehicles : undefined,
      })
      setDispatchResult(result)
      setScenarios(dispatchToScenarios(result, riskSettings))
    } catch (err) {
      setDispatchError(err instanceof Error ? err.message : String(err))
    } finally {
      setDispatchLoading(false)
    }
  }, [activeWarehouseId, analysisDateTime, incomingVehicles, riskSettings])

  // Keep a ref to latest handleDispatch to avoid stale closures in effects
  const handleDispatchRef = useRef(handleDispatch)
  useEffect(() => { handleDispatchRef.current = handleDispatch }, [handleDispatch])

  // Auto-re-dispatch when datetime changes (only if a warehouse is already selected)
  const isFirstDateTimeRender = useRef(true)
  useEffect(() => {
    if (isFirstDateTimeRender.current) { isFirstDateTimeRender.current = false; return }
    if (activeWarehouseId) handleDispatchRef.current()
  }, [analysisDateTime]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Page header */}
      <div className="px-4 py-3 border-b border-border bg-surface shrink-0 flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold text-foreground">Оптимизатор транспортных вызовов</h1>
          <p className="text-xs text-muted mt-0.5">
            Рекомендации на основе ML-прогноза. Подтверждайте или корректируйте вызовы.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {warehouseName && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted">Склад:</span>
              <span className="text-xs font-semibold text-accent bg-accent/10 rounded px-2 py-1">
                {warehouseName}
              </span>
              <button
                onClick={() => { setWarehouseFilter(''); setSelectedWarehouseId(null); setDispatchResult(null) }}
                className="text-xs text-muted hover:text-foreground transition-colors"
              >
                ✕
              </button>
            </div>
          )}
          <button
            onClick={handleDispatch}
            disabled={!activeWarehouseId || dispatchLoading}
            className="px-3 py-1.5 text-xs font-semibold rounded bg-accent text-white hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {dispatchLoading ? 'Расчёт…' : 'Получить план развозки'}
          </button>
        </div>
      </div>

      {/* Dispatch result panel */}
      {(dispatchResult || dispatchError) && (
        <div className="px-4 py-3 border-b border-border bg-elevated shrink-0">
          {dispatchError && (
            <p className="text-xs text-status-red">{dispatchError}</p>
          )}
          {dispatchResult && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-foreground">
                  План развозки · {dispatchResult.routes.length} маршрутов · итого:{' '}
                  <span className="text-accent">{dispatchResult.total_cost.toLocaleString('ru-RU')} ₽</span>
                </span>
                <button
                  onClick={() => setDispatchResult(null)}
                  className="text-xs text-muted hover:text-foreground"
                >
                  ✕ Скрыть
                </button>
              </div>
              <div className="overflow-x-auto max-h-56 overflow-y-auto rounded border border-border">
                <table className="w-full text-xs">
                  <thead className="bg-surface sticky top-0">
                    <tr>
                      <th className="text-left px-2 py-1.5 text-muted font-medium">Маршрут</th>
                      <th className="text-left px-2 py-1.5 text-muted font-medium">Горизонт</th>
                      <th className="text-left px-2 py-1.5 text-muted font-medium">ТС</th>
                      <th className="text-right px-2 py-1.5 text-muted font-medium">Кол-во</th>
                      <th className="text-right px-2 py-1.5 text-muted font-medium">Спрос</th>
                      <th className="text-right px-2 py-1.5 text-muted font-medium">Отправлено</th>
                      <th className="text-right px-2 py-1.5 text-muted font-medium">Стоимость ₽</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dispatchResult.routes.flatMap(rp =>
                      rp.plan.map((row, i) => (
                        <tr key={`${rp.route_id}-${i}`} className="border-t border-border hover:bg-surface/50">
                          <td className="px-2 py-1 font-mono text-muted">{row.route_id}</td>
                          <td className="px-2 py-1 text-muted">{HORIZON_LABELS[row.horizon] ?? row.horizon}</td>
                          <td className="px-2 py-1">{row.vehicle_type === 'none' ? '—' : row.vehicle_type}</td>
                          <td className="px-2 py-1 text-right font-mono">{row.vehicles_count || '—'}</td>
                          <td className="px-2 py-1 text-right font-mono">{row.demand_new.toFixed(0)}</td>
                          <td className={`px-2 py-1 text-right font-mono ${row.actually_shipped >= row.demand_new ? 'text-status-green' : 'text-status-yellow'}`}>
                            {row.actually_shipped.toFixed(0)}
                          </td>
                          <td className="px-2 py-1 text-right font-mono">{row.cost_total.toLocaleString('ru-RU')}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Two-column layout */}
      <div className="flex flex-1 overflow-hidden gap-4 p-4">
        {/* Left: table (2/3) */}
        <div className="flex-[2] min-w-0 overflow-y-auto">
          <RecommendationTable
            recommendations={displayRecs}
            selectedId={selectedRec?.id ?? null}
            onSelect={handleSelect}
            onCall={handleCall}
            warehouseFilter={activeWarehouseId || undefined}
            warehouseLabelById={warehouseLabelById}
          />
        </div>

        {/* Right: cost-benefit (1/3) */}
        <div className="flex-1 min-w-[300px] overflow-hidden">
          <CostBenefitCard recommendation={selectedRec} scenarios={scenarios} />
        </div>
      </div>
    </div>
  )
}
