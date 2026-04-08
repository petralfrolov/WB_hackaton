import { useState, useEffect, useCallback, useMemo } from 'react'
import { getAvailableDates, postRetrospectiveMetrics, getWarehouses } from '../api'
import type {
  ApiMetricsResponse,
  ApiAvailableDates,
  ApiWarehouseInfo,
} from '../types'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card'
import {
  ForecastVsActualChart,
  KpiCards,
  MetricsTimeSeriesChart,
  RouteComparisonTable,
} from '../components/metrics'
import { cn } from '../lib/utils'

const HORIZONS = [
  { value: 'B: +2h', label: '+2 ч' },
  { value: 'C: +4h', label: '+4 ч' },
  { value: 'D: +6h', label: '+6 ч' },
]

export function MetricsPage() {
  const [allWarehouses, setAllWarehouses] = useState<ApiWarehouseInfo[]>([])
  const [warehouseId, setWarehouseId] = useState('')
  const [horizon, setHorizon] = useState('B: +2h')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [availableDates, setAvailableDates] = useState<ApiAvailableDates | null>(null)
  const [data, setData] = useState<ApiMetricsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null)

  // Load warehouse list
  useEffect(() => {
    getWarehouses().then(setAllWarehouses).catch(() => {})
  }, [])

  // Auto-select first warehouse
  useEffect(() => {
    if (!warehouseId && allWarehouses.length > 0) {
      setWarehouseId(allWarehouses[0].id)
    }
  }, [allWarehouses, warehouseId])

  // Load available dates when warehouse changes
  useEffect(() => {
    if (!warehouseId) return
    setAvailableDates(null)
    getAvailableDates(warehouseId)
      .then(ad => {
        setAvailableDates(ad)
        if (ad.min_date && ad.max_date) {
          setDateFrom(ad.min_date.slice(0, 16))
          setDateTo(ad.max_date.slice(0, 16))
        }
      })
      .catch(() => {})
  }, [warehouseId])

  // Fetch metrics
  const fetchMetrics = useCallback(async () => {
    if (!warehouseId || !dateFrom || !dateTo) return
    setLoading(true)
    setError(null)
    try {
      const resp = await postRetrospectiveMetrics({
        warehouse_id: warehouseId,
        date_from: dateFrom,
        date_to: dateTo,
        horizon,
      })
      setData(resp)
      // Auto-select first route
      if (resp.route_summary.length > 0 && !selectedRouteId) {
        setSelectedRouteId(resp.route_summary[0].route_id)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Ошибка загрузки метрик')
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [warehouseId, dateFrom, dateTo, horizon, selectedRouteId])

  // Auto-fetch when filters are ready
  useEffect(() => {
    if (warehouseId && dateFrom && dateTo) {
      fetchMetrics()
    }
  }, [warehouseId, dateFrom, dateTo, horizon])

  // Chart data for selected route
  const routeChartData = useMemo(() => {
    if (!data || !selectedRouteId) return []
    return data.forecast_vs_actual
      .filter(fva => fva.route_id === selectedRouteId)
      .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
  }, [data, selectedRouteId])

  // Available route IDs from data
  const routeIds = useMemo(() => {
    if (!data) return []
    return data.route_summary.map(r => r.route_id)
  }, [data])

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Filter bar ─────────────────────────────────────────────────── */}
      <div className="shrink-0 border-b border-border bg-surface px-4 py-3 flex flex-wrap items-end gap-4">
        {/* Warehouse */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-muted uppercase tracking-widest">Склад</label>
          <select
            value={warehouseId}
            onChange={e => { setWarehouseId(e.target.value); setData(null); setSelectedRouteId(null) }}
            className="h-8 rounded bg-elevated border border-border px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-accent min-w-[180px]"
          >
            <option value="">Выберите склад</option>
            {allWarehouses.map(w => (
              <option key={w.id} value={w.id}>{w.name} ({w.city})</option>
            ))}
          </select>
        </div>

        {/* Date from */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-muted uppercase tracking-widest">Дата с</label>
          <input
            type="datetime-local"
            value={dateFrom}
            onChange={e => setDateFrom(e.target.value)}
            className="h-8 rounded bg-elevated border border-border px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>

        {/* Date to */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-muted uppercase tracking-widest">Дата по</label>
          <input
            type="datetime-local"
            value={dateTo}
            onChange={e => setDateTo(e.target.value)}
            className="h-8 rounded bg-elevated border border-border px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>

        {/* Horizon */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-muted uppercase tracking-widest">Горизонт</label>
          <div className="flex gap-0.5">
            {HORIZONS.map(h => (
              <button
                key={h.value}
                onClick={() => setHorizon(h.value)}
                className={cn(
                  'h-8 px-3 rounded text-xs font-medium transition-colors border',
                  horizon === h.value
                    ? 'bg-accent text-background border-accent'
                    : 'bg-elevated border-border text-muted hover:text-foreground',
                )}
              >
                {h.label}
              </button>
            ))}
          </div>
        </div>

        {/* Fetch button */}
        <button
          onClick={fetchMetrics}
          disabled={loading || !warehouseId || !dateFrom || !dateTo}
          className="h-8 px-4 rounded bg-accent text-background text-xs font-semibold hover:bg-accent/90 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Загрузка…' : 'Обновить'}
        </button>

        {availableDates && availableDates.count > 0 && (
          <span className="text-[10px] text-muted self-end">
            {availableDates.count} расчётов в базе
          </span>
        )}
      </div>

      {/* ── Content ────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {error && (
          <div className="text-sm text-status-red bg-status-red/10 border border-status-red/30 rounded px-4 py-3">
            {error}
          </div>
        )}

        {!data && !loading && !error && (
          <div className="text-sm text-muted text-center py-12">
            Выберите склад и диапазон дат для просмотра метрик
          </div>
        )}

        {loading && (
          <div className="text-sm text-muted text-center py-12">Загрузка данных…</div>
        )}

        {data && (
          <>
            {/* Section B: KPI Cards */}
            <KpiCards aggregate={data.aggregate} />

            {/* Section A: Forecast vs Actual */}
            <Card>
              <CardHeader>
                <CardTitle>Прогноз vs Факт</CardTitle>
                <select
                  value={selectedRouteId || ''}
                  onChange={e => setSelectedRouteId(e.target.value)}
                  className="h-7 rounded bg-elevated border border-border px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
                >
                  {routeIds.map(rid => (
                    <option key={rid} value={rid}>Маршрут {rid}</option>
                  ))}
                </select>
              </CardHeader>
              <CardContent>
                {routeChartData.length > 0 ? (
                  <ForecastVsActualChart data={routeChartData} />
                ) : (
                  <div className="text-xs text-muted py-8 text-center">
                    Нет данных для выбранного маршрута
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Section C: Metrics time series */}
            <Card>
              <CardHeader>
                <CardTitle>Метрики во времени</CardTitle>
              </CardHeader>
              <CardContent>
                <MetricsTimeSeriesChart data={data.time_series} />
              </CardContent>
            </Card>

            {/* Section D: Route comparison table */}
            <Card>
              <CardHeader>
                <CardTitle>Сравнение маршрутов</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <RouteComparisonTable
                  routes={data.route_summary}
                  selectedRouteId={selectedRouteId}
                  onSelectRoute={setSelectedRouteId}
                />
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
