import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'
import type { ApiWarehouseMetrics } from '../../types'
import { horizonDisplayLabel } from '../../lib/utils'

type MetricKey = 'p_cover' | 'fill_rate' | 'cpo' | 'fleet_utilization' | 'capacity_shortfall'

interface Props {
  metricKey: MetricKey | null
  metrics: ApiWarehouseMetrics
  onClose: () => void
}

/* ── tiny shared table styles ─────────────────────────────────────────────── */
const thCls = 'px-3 py-1.5 text-left text-[11px] font-semibold text-muted uppercase tracking-wide border-b border-border/60'
const tdCls = 'px-3 py-1.5 text-[12px] font-mono border-b border-border/30'
const sumCls = 'px-3 py-1.5 text-[12px] font-mono font-bold border-t border-border/60'

function PCoverDetail({ m }: { m: ApiWarehouseMetrics }) {
  const rows = m.p_cover_detail ?? []
  return (
    <>
      <div className="text-[11px] text-muted mb-2">
        p_cover = min(p по горизонтам B…D) = <span className="text-white font-bold">{(m.p_cover * 100).toFixed(1)}%</span>
      </div>
      <table className="w-full text-left">
        <thead><tr>
          <th className={thCls}>Горизонт</th>
          <th className={thCls}>Вместимость отпр.</th>
          <th className={thCls}>Спрос (точеч.)</th>
          <th className={thCls}>Запас неопр.</th>
          <th className={thCls}>P покрытия</th>
        </tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="hover:bg-white/5">
              <td className={tdCls}>{horizonDisplayLabel(r.horizon)}</td>
              <td className={tdCls}>{r.capacity.toFixed(0)}</td>
              <td className={tdCls}>{r.demand.toFixed(0)}</td>
              <td className={tdCls}>{r.margin.toFixed(0)}</td>
              <td className={tdCls}>
                <span className={r.p_cover >= 0.9 ? 'text-status-green' : r.p_cover >= 0.7 ? 'text-status-yellow' : 'text-status-red'}>
                  {(r.p_cover * 100).toFixed(1)}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-[10px] text-muted mt-2">
        Формула: z<sub>t</sub> = (вместимость − спрос) / (запас / z<sub>α</sub>), p<sub>t</sub> = Φ(z<sub>t</sub>)
      </div>
    </>
  )
}

function FillRateDetail({ m }: { m: ApiWarehouseMetrics }) {
  const rows = m.fill_rate_detail ?? []
  const totalShipped = rows.reduce((s, r) => s + r.shipped, 0)
  const totalCap = rows.reduce((s, r) => s + r.capacity_sent, 0)
  return (
    <>
      <div className="text-[11px] text-muted mb-2">
        fill_rate = отправлено / вместимость = <span className="text-white font-bold">{(m.fill_rate * 100).toFixed(1)}%</span>
      </div>
      <table className="w-full text-left">
        <thead><tr>
          <th className={thCls}>Маршрут</th>
          <th className={thCls}>Отправлено</th>
          <th className={thCls}>Вместимость</th>
          <th className={thCls}>Загрузка</th>
        </tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="hover:bg-white/5">
              <td className={tdCls}>{r.route_id}</td>
              <td className={tdCls}>{r.shipped.toFixed(0)}</td>
              <td className={tdCls}>{r.capacity_sent.toFixed(0)}</td>
              <td className={tdCls}>
                <span className={r.fill_rate >= 0.8 ? 'text-status-green' : r.fill_rate >= 0.5 ? 'text-status-yellow' : 'text-status-red'}>
                  {(r.fill_rate * 100).toFixed(1)}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot><tr className="bg-white/5">
          <td className={sumCls}>Итого</td>
          <td className={sumCls}>{totalShipped.toFixed(0)}</td>
          <td className={sumCls}>{totalCap.toFixed(0)}</td>
          <td className={sumCls}>
            <span className={m.fill_rate >= 0.8 ? 'text-status-green' : m.fill_rate >= 0.5 ? 'text-status-yellow' : 'text-status-red'}>
              {(m.fill_rate * 100).toFixed(1)}%
            </span>
          </td>
        </tr></tfoot>
      </table>
    </>
  )
}

function CpoDetail({ m }: { m: ApiWarehouseMetrics }) {
  const rows = m.cpo_detail ?? []
  const totalCost = rows.reduce((s, r) => s + r.cost, 0)
  const totalShipped = rows.reduce((s, r) => s + r.shipped, 0)
  return (
    <>
      <div className="text-[11px] text-muted mb-2">
        CPO = общие затраты / ед. отправлено = <span className="text-white font-bold">{m.cpo.toFixed(0)} ₽</span>
      </div>
      <table className="w-full text-left">
        <thead><tr>
          <th className={thCls}>Маршрут</th>
          <th className={thCls}>Затраты, ₽</th>
          <th className={thCls}>Отправлено</th>
          <th className={thCls}>CPO, ₽</th>
        </tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="hover:bg-white/5">
              <td className={tdCls}>{r.route_id}</td>
              <td className={tdCls}>{r.cost.toFixed(0)}</td>
              <td className={tdCls}>{r.shipped.toFixed(0)}</td>
              <td className={tdCls}>{r.cpo.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot><tr className="bg-white/5">
          <td className={sumCls}>Итого</td>
          <td className={sumCls}>{totalCost.toFixed(0)}</td>
          <td className={sumCls}>{totalShipped.toFixed(0)}</td>
          <td className={sumCls}>{m.cpo.toFixed(0)}</td>
        </tr></tfoot>
      </table>
    </>
  )
}

function FleetUtilDetail({ m }: { m: ApiWarehouseMetrics }) {
  const fleet = m.fleet_detail ?? []
  const dispatched = m.dispatched_detail ?? []
  const utilRows = m.utilization_detail ?? []
  return (
    <>
      <div className="text-[11px] text-muted mb-2">
        Утилизация = потребность (спрос + запас) / доступный флот ={' '}
        <span className="text-white font-bold">{m.fleet_utilization_ratio != null ? m.fleet_utilization_ratio.toFixed(2) : '—'}</span>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="bg-white/5 rounded px-3 py-2">
          <div className="text-[10px] text-muted">Спрос (точечный)</div>
          <div className="text-sm font-bold font-mono">{m.total_demand_units?.toFixed(0) ?? '—'}</div>
        </div>
        <div className="bg-white/5 rounded px-3 py-2">
          <div className="text-[10px] text-muted">+ Запас неопределённости</div>
          <div className="text-sm font-bold font-mono">{m.total_conformal_margin?.toFixed(0) ?? '—'}</div>
        </div>
        <div className="bg-white/5 rounded px-3 py-2">
          <div className="text-[10px] text-muted">= Потребность итого</div>
          <div className="text-sm font-bold font-mono">{m.required_capacity_units?.toFixed(0) ?? '—'}</div>
        </div>
      </div>

      {/* Available fleet */}
      <div className="text-[11px] text-muted mb-1 mt-3">Доступный флот (к горизонту +6ч)</div>
      <table className="w-full text-left mb-3">
        <thead><tr>
          <th className={thCls}>Тип ТС</th>
          <th className={thCls}>Сейчас</th>
          <th className={thCls}>К +6ч</th>
          <th className={thCls}>Вместимость</th>
          <th className={thCls}>Итого (+6ч)</th>
        </tr></thead>
        <tbody>
          {fleet.map((r, i) => (
            <tr key={i} className="hover:bg-white/5">
              <td className={tdCls}>{r.vehicle_type}</td>
              <td className={tdCls}>{r.available}</td>
              <td className={tdCls}>{r.available_at_6h ?? r.available}</td>
              <td className={tdCls}>{r.capacity_units}</td>
              <td className={tdCls}>{r.total_capacity.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot><tr className="bg-white/5">
          <td className={sumCls}>Итого</td>
          <td className={sumCls}>{fleet.reduce((s, r) => s + r.available, 0)}</td>
          <td className={sumCls}>{fleet.reduce((s, r) => s + (r.available_at_6h ?? r.available), 0)}</td>
          <td className={sumCls}></td>
          <td className={sumCls}>{m.available_capacity_units?.toFixed(0) ?? '—'}</td>
        </tr></tfoot>
      </table>

      {/* Dispatched */}
      <div className="text-[11px] text-muted mb-1">Фактически отправлено (MILP решение)</div>
      <table className="w-full text-left mb-3">
        <thead><tr>
          <th className={thCls}>Тип ТС</th>
          <th className={thCls}>Кол-во</th>
          <th className={thCls}>Вместимость</th>
          <th className={thCls}>Итого</th>
        </tr></thead>
        <tbody>
          {dispatched.map((r, i) => (
            <tr key={i} className="hover:bg-white/5">
              <td className={tdCls}>{r.vehicle_type}</td>
              <td className={tdCls}>{r.vehicles_count.toFixed(1)}</td>
              <td className={tdCls}>{r.capacity_units}</td>
              <td className={tdCls}>{r.total_capacity.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot><tr className="bg-white/5">
          <td className={sumCls}>Итого</td>
          <td className={sumCls}></td>
          <td className={sumCls}></td>
          <td className={sumCls}>{m.dispatched_capacity_units?.toFixed(0) ?? '—'}</td>
        </tr></tfoot>
      </table>

      {/* Per-horizon demand vs dispatched */}
      <div className="text-[11px] text-muted mb-1">Детализация по горизонтам</div>
      <table className="w-full text-left">
        <thead><tr>
          <th className={thCls}>Горизонт</th>
          <th className={thCls}>Спрос</th>
          <th className={thCls}>Запас</th>
          <th className={thCls}>Потребность</th>
          <th className={thCls}>Отправлено</th>
          <th className={thCls}>Δ</th>
        </tr></thead>
        <tbody>
          {utilRows.map((r, i) => {
            const delta = r.dispatched_capacity - r.demand_with_margin
            return (
              <tr key={i} className="hover:bg-white/5">
                <td className={tdCls}>{horizonDisplayLabel(r.horizon)}</td>
                <td className={tdCls}>{r.demand.toFixed(0)}</td>
                <td className={tdCls}>{r.margin.toFixed(0)}</td>
                <td className={tdCls}>{r.demand_with_margin.toFixed(0)}</td>
                <td className={tdCls}>{r.dispatched_capacity.toFixed(0)}</td>
                <td className={`${tdCls} ${delta >= 0 ? 'text-status-green' : 'text-status-red'}`}>
                  {delta >= 0 ? '+' : ''}{delta.toFixed(0)}
                </td>
              </tr>
            )
          })}
        </tbody>
        <tfoot><tr className="bg-white/5">
          <td className={sumCls}>Итого</td>
          <td className={sumCls}>{utilRows.reduce((s, r) => s + r.demand, 0).toFixed(0)}</td>
          <td className={sumCls}>{utilRows.reduce((s, r) => s + r.margin, 0).toFixed(0)}</td>
          <td className={sumCls}>{utilRows.reduce((s, r) => s + r.demand_with_margin, 0).toFixed(0)}</td>
          <td className={sumCls}>{utilRows.reduce((s, r) => s + r.dispatched_capacity, 0).toFixed(0)}</td>
          {(() => {
            const totalDelta = utilRows.reduce((s, r) => s + (r.dispatched_capacity - r.demand_with_margin), 0)
            return <td className={`${sumCls} ${totalDelta >= 0 ? 'text-status-green' : 'text-status-red'}`}>
              {totalDelta >= 0 ? '+' : ''}{totalDelta.toFixed(0)}
            </td>
          })()}
        </tr></tfoot>
      </table>
    </>
  )
}

function CapacityShortfallDetail({ m }: { m: ApiWarehouseMetrics }) {
  const shortfall = m.fleet_capacity_shortfall
  return (
    <>
      <div className="text-[11px] text-muted mb-2">
        Нехватка = потребность − доступный флот ={' '}
        <span className={`font-bold ${shortfall != null && shortfall > 0 ? 'text-status-red' : 'text-status-green'}`}>
          {shortfall != null ? (shortfall > 0 ? `+${shortfall.toFixed(0)}` : shortfall.toFixed(0)) : '—'} ед.
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="bg-white/5 rounded px-3 py-2">
          <div className="text-[10px] text-muted">Потребность</div>
          <div className="text-sm font-bold font-mono">{m.required_capacity_units?.toFixed(0) ?? '—'}</div>
          <div className="text-[9px] text-muted">спрос {m.total_demand_units?.toFixed(0) ?? '?'} + запас {m.total_conformal_margin?.toFixed(0) ?? '?'}</div>
        </div>
        <div className="bg-white/5 rounded px-3 py-2">
          <div className="text-[10px] text-muted">Доступный флот</div>
          <div className="text-sm font-bold font-mono">{m.available_capacity_units?.toFixed(0) ?? '—'}</div>
        </div>
        <div className="bg-white/5 rounded px-3 py-2">
          <div className="text-[10px] text-muted">Фактически отправлено</div>
          <div className="text-sm font-bold font-mono">{m.dispatched_capacity_units?.toFixed(0) ?? '—'}</div>
        </div>
      </div>

      {/* Reuse fleet utilization horizon detail */}
      <FleetUtilDetail m={m} />
    </>
  )
}

const METRIC_TITLES: Record<MetricKey, string> = {
  p_cover: 'Вероятность покрытия спроса',
  fill_rate: 'Коэффициент загрузки ТС',
  cpo: 'Стоимость одной доставки (CPO)',
  fleet_utilization: 'Коэффициент утилизации флота',
  capacity_shortfall: 'Нехватка вместимости',
}

export function MetricDetailModal({ metricKey, metrics, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!metricKey) return
    const handle = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [metricKey, onClose])

  useEffect(() => {
    if (!metricKey) return
    const handle = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [metricKey, onClose])

  if (!metricKey) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div
        ref={ref}
        className="relative bg-surface border border-border rounded-xl shadow-2xl w-[700px] max-w-[90vw] max-h-[80vh] overflow-y-auto p-5"
      >
        <button onClick={onClose} className="absolute top-3 right-3 text-muted hover:text-white transition-colors">
          <X className="w-5 h-5" />
        </button>
        <h3 className="text-base font-semibold mb-4 pr-8">{METRIC_TITLES[metricKey]}</h3>
        {metricKey === 'p_cover' && <PCoverDetail m={metrics} />}
        {metricKey === 'fill_rate' && <FillRateDetail m={metrics} />}
        {metricKey === 'cpo' && <CpoDetail m={metrics} />}
        {metricKey === 'fleet_utilization' && <FleetUtilDetail m={metrics} />}
        {metricKey === 'capacity_shortfall' && <CapacityShortfallDetail m={metrics} />}
      </div>
    </div>
  )
}
