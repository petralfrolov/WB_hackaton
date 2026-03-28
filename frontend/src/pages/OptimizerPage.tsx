import { useState, useCallback, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { TransportRecommendation, CostScenario } from '../types'
import { recommendations as initialRecs, getCostScenarios, warehouses } from '../data/mockData'
import { RecommendationTable } from '../components/optimizer/RecommendationTable'
import { CostBenefitCard } from '../components/optimizer/CostBenefitCard'

export function OptimizerPage() {
  const [searchParams] = useSearchParams()
  const [recs, setRecs] = useState<TransportRecommendation[]>(initialRecs)
  const [selectedRec, setSelectedRec] = useState<TransportRecommendation | null>(null)
  const [scenarios, setScenarios] = useState<CostScenario[]>([])
  const [warehouseFilter, setWarehouseFilter] = useState<string>(
    searchParams.get('warehouseId') ?? '',
  )

  // Sync URL param when it changes (e.g. navigation from drawer)
  useEffect(() => {
    const id = searchParams.get('warehouseId') ?? ''
    setWarehouseFilter(id)
  }, [searchParams])

  const warehouseName = warehouseFilter
    ? warehouses.find(w => w.id === warehouseFilter)?.name ?? warehouseFilter
    : null
  const warehouseLabelById = Object.fromEntries(warehouses.map(w => [w.id, w.name]))

  const handleSelect = useCallback((rec: TransportRecommendation) => {
    setSelectedRec(rec)
    setScenarios(getCostScenarios(rec.id))
  }, [])

  const handleCall = useCallback((id: string) => {
    setRecs(prev =>
      prev.map(r =>
        r.id === id ? { ...r, status: 'called' as const } : r,
      ),
    )
    setSelectedRec(prev => (prev?.id === id ? { ...prev, status: 'called' as const } : prev))
  }, [])

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
        {warehouseName && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">Склад:</span>
            <span className="text-xs font-semibold text-accent bg-accent/10 rounded px-2 py-1">
              {warehouseName}
            </span>
            <button
              onClick={() => setWarehouseFilter('')}
              className="text-xs text-muted hover:text-foreground transition-colors ml-1"
            >
              ✕ Сбросить
            </button>
          </div>
        )}
      </div>

      {/* Two-column layout */}
      <div className="flex flex-1 overflow-hidden gap-4 p-4">
        {/* Left: table (2/3) */}
        <div className="flex-[2] min-w-0 overflow-y-auto">
          <RecommendationTable
            recommendations={recs}
            selectedId={selectedRec?.id ?? null}
            onSelect={handleSelect}
            onCall={handleCall}
            warehouseFilter={warehouseFilter || undefined}
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
