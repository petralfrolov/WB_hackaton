import { useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  LabelList,
} from 'recharts'
import type { TransportRecommendation, CostScenario } from '../../types'
import { computeCostFromBreakdown } from '../../data/mockData'
import { fmtCurrency, fmt } from '../../lib/utils'
import { Clock, TrendingDown, X, Calculator } from 'lucide-react'

interface CostBenefitCardProps {
  recommendation: TransportRecommendation | null
  scenarios: CostScenario[]
}

interface TooltipPayloadEntry {
  value: number
  payload: CostScenario
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayloadEntry[]
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null
  const sc = payload[0].payload
  return (
    <div className="bg-elevated border border-border rounded px-3 py-2 text-xs shadow-lg">
      <div className="font-semibold text-foreground">{sc.name}</div>
      <div className="text-muted mt-0.5">{sc.description}</div>
      <div className="mt-1.5 flex gap-3">
        <span className="text-status-green font-mono font-bold">{fmtCurrency(sc.cost)}</span>
        <span className="text-muted flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {sc.time}
        </span>
      </div>
    </div>
  )
}

// ── Formula breakdown modal ──────────────────────────────────────────────────

function FormulaModal({ scenario, onClose }: { scenario: CostScenario; onClose: () => void }) {
  const b = scenario.breakdown
  const fixedTotal = b.vehicles.reduce((s, v) => s + v.fixedCost * v.count, 0)
  const emptyPenaltyTotal = b.wEcon * b.vehicles.reduce((s, v) => s + (v.capacity - v.load) * v.count, 0) * b.pEmpty
  const delayPenaltyTotal = b.wUrg * b.itemsWaiting * b.avgWaitMinutes * b.pDelay
  const total = computeCostFromBreakdown(b)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      {/* Panel */}
      <div className="relative bg-surface border border-border rounded-xl shadow-2xl w-[560px] max-h-[90vh] overflow-y-auto z-10">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div>
            <div className="flex items-center gap-2">
              <Calculator className="w-4 h-4 text-accent" />
              <span className="font-semibold text-foreground">Формула стоимости</span>
            </div>
            <div className="text-xs text-muted mt-0.5">{scenario.name} · {scenario.description}</div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-elevated text-muted hover:text-foreground transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-5">
          {/* Formula display */}
          <div className="bg-elevated rounded-lg px-4 py-3 font-mono text-xs text-muted leading-relaxed overflow-x-auto">
            <div>J = Σ(C<sub>fixed,i</sub> × N<sub>i</sub>)</div>
            <div className="ml-4">+ W<sub>econ</sub> × Σ(Cap<sub>i</sub> − L<sub>i</sub>) × P<sub>empty</sub></div>
            <div className="ml-4">+ W<sub>urg</sub> × m × T<sub>wait</sub> × P<sub>delay</sub></div>
          </div>

          {/* Term 1 — Fixed costs */}
          <section>
            <div className="section-label mb-2 flex items-center gap-2">
              <span className="w-4 h-4 rounded-sm bg-status-green/20 text-status-green text-[10px] flex items-center justify-center font-bold">1</span>
              Постоянные затраты — Σ(C<sub>fixed,i</sub> × N<sub>i</sub>)
            </div>
            <div className="bg-elevated rounded-lg overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-muted">
                    <th className="px-3 py-2 text-left">ТС</th>
                    <th className="px-3 py-2 text-right">Кол-во (N)</th>
                    <th className="px-3 py-2 text-right">C<sub>fixed</sub> / ед.</th>
                    <th className="px-3 py-2 text-right">Итого</th>
                  </tr>
                </thead>
                <tbody>
                  {b.vehicles.map((v, i) => (
                    <tr key={i} className="border-b border-border/40 last:border-0">
                      <td className="px-3 py-1.5 text-foreground">{v.name}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{v.count}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{fmtCurrency(v.fixedCost)}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-foreground">{fmtCurrency(v.fixedCost * v.count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="text-right text-xs font-mono text-status-green mt-1 pr-1">
              = {fmtCurrency(fixedTotal)}
            </div>
          </section>

          {/* Term 2 — Empty slot penalty */}
          <section>
            <div className="section-label mb-2 flex items-center gap-2">
              <span className="w-4 h-4 rounded-sm bg-accent/20 text-accent text-[10px] flex items-center justify-center font-bold">2</span>
              Штраф за недозагрузку — W<sub>econ</sub> × Σ(Cap<sub>i</sub> − L<sub>i</sub>) × P<sub>empty</sub>
            </div>
            <div className="bg-elevated rounded-lg px-4 py-3 space-y-1.5 text-xs">
              <div className="grid grid-cols-2 gap-2 text-muted">
                {b.vehicles.map((v, i) => (
                  <div key={i} className="flex justify-between">
                    <span>{v.name}: Cap={fmt(v.capacity)}, L={fmt(v.load)}</span>
                    <span className="font-mono text-foreground">Δ={fmt((v.capacity - v.load) * v.count)}</span>
                  </div>
                ))}
              </div>
              <div className="border-t border-border/50 pt-2 flex items-center gap-2 flex-wrap">
                <span className="text-muted">W<sub>econ</sub> = {b.wEcon.toFixed(2)}</span>
                <span className="text-muted">P<sub>empty</sub> = {fmtCurrency(b.pEmpty)}/ед.</span>
              </div>
            </div>
            <div className="text-right text-xs font-mono text-accent mt-1 pr-1">
              = {fmtCurrency(emptyPenaltyTotal)}
            </div>
          </section>

          {/* Term 3 — Delay penalty */}
          <section>
            <div className="section-label mb-2 flex items-center gap-2">
              <span className="w-4 h-4 rounded-sm bg-status-yellow/20 text-status-yellow text-[10px] flex items-center justify-center font-bold">3</span>
              Штраф за ожидание — W<sub>urg</sub> × m × T<sub>wait</sub> × P<sub>delay</sub>
            </div>
            <div className="bg-elevated rounded-lg px-4 py-3 text-xs space-y-1">
              <div className="flex items-center gap-4 flex-wrap">
                <span className="text-muted">W<sub>urg</sub> = {b.wUrg.toFixed(2)}</span>
                <span className="text-muted">m = {b.itemsWaiting} поз.</span>
                <span className="text-muted">T<sub>wait</sub> = {b.avgWaitMinutes} мин.</span>
                <span className="text-muted">P<sub>delay</sub> = {fmtCurrency(b.pDelay)}/мин.</span>
              </div>
            </div>
            <div className="text-right text-xs font-mono text-status-yellow mt-1 pr-1">
              = {fmtCurrency(delayPenaltyTotal)}
            </div>
          </section>

          {/* Total */}
          <div className="bg-status-green/10 border border-status-green/30 rounded-lg px-4 py-3 flex items-center justify-between">
            <span className="text-sm font-semibold text-foreground">J итого</span>
            <div className="text-right">
              <div className="text-xs text-muted font-mono">
                {fmtCurrency(fixedTotal)} + {fmtCurrency(emptyPenaltyTotal)} + {fmtCurrency(delayPenaltyTotal)}
              </div>
              <div className="text-xl font-bold font-mono text-status-green">{fmtCurrency(total)}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

const SCENARIO_COLORS = ['#3FB950', '#58A6FF', '#D29922']

export function CostBenefitCard({ recommendation, scenarios }: CostBenefitCardProps) {
  const [modalScenario, setModalScenario] = useState<CostScenario | null>(null)

  if (!recommendation) {
    return (
      <div className="bg-surface border border-border rounded-lg h-full flex items-center justify-center">
        <div className="text-center text-muted px-8">
          <TrendingDown className="w-8 h-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Выберите маршрут в таблице слева</p>
          <p className="text-xs mt-1 opacity-60">для анализа стоимости сценариев</p>
        </div>
      </div>
    )
  }

  const cheapest = scenarios[0]
  const mostExpensive = scenarios[scenarios.length - 1]
  const savings = mostExpensive.cost - cheapest.cost

  const chartData = scenarios.map(sc => ({
    ...sc,
    cost: sc.cost,
  }))

  return (
    <div className="bg-surface border border-border rounded-lg flex flex-col overflow-hidden h-full">
      {/* Formula modal */}
      {modalScenario && (
        <FormulaModal scenario={modalScenario} onClose={() => setModalScenario(null)} />
      )}
      {/* Header */}
      <div className="px-4 py-3 border-b border-border shrink-0">
        <div className="section-label mb-0.5">Анализ стоимости</div>
        <div className="text-sm font-semibold text-foreground">{recommendation.route}</div>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1 overflow-y-auto">
        {/* Savings banner */}
        <div className="bg-status-green/10 border border-status-green/25 rounded-lg px-4 py-3 flex items-center gap-3">
          <TrendingDown className="w-5 h-5 text-status-green shrink-0" />
          <div>
            <div className="text-xs text-muted">Экономия vs стандарт</div>
            <div className="text-xl font-bold font-mono text-status-green">
              {fmtCurrency(savings)}
            </div>
          </div>
          <div className="ml-auto text-right">
            <div className="text-xs text-muted">Лучший вариант</div>
            <div className="text-sm font-semibold text-foreground">{cheapest.name}</div>
          </div>
        </div>

        {/* Bar chart */}
        <div>
          <div className="section-label mb-2">Сравнение сценариев</div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart
              layout="vertical"
              data={chartData}
              margin={{ top: 0, right: 70, bottom: 0, left: 0 }}
            >
              <CartesianGrid horizontal={false} stroke="#30363D" strokeDasharray="3 3" />
              <XAxis
                type="number"
                tick={{ fill: '#7D8590', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => `${fmt(v as number)}₽`}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={88}
                tick={{ fill: '#E6EDF3', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="cost" radius={[0, 3, 3, 0]} maxBarSize={28}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={SCENARIO_COLORS[i] ?? '#58A6FF'} fillOpacity={0.85} />
                ))}
                <LabelList
                  dataKey="cost"
                  position="right"
                  formatter={(v: number) => fmtCurrency(v)}
                  style={{ fill: '#E6EDF3', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Detail cards */}
        <div className="space-y-2">
          {scenarios.map((sc, i) => (
            <button
              key={sc.id}
              onClick={() => setModalScenario(sc)}
              className="w-full bg-elevated rounded-lg px-3 py-2.5 flex items-center gap-3 hover:bg-border/60 transition-colors text-left group"
              title="Открыть разбивку формулы"
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: SCENARIO_COLORS[i] ?? '#58A6FF' }}
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-foreground">{sc.name}</div>
                <div className="text-xs text-muted truncate">{sc.description}</div>
              </div>
              <div className="text-right shrink-0">
                <div
                  className="text-sm font-mono font-bold"
                  style={{ color: SCENARIO_COLORS[i] ?? '#58A6FF' }}
                >
                  {fmtCurrency(sc.cost)}
                </div>
                <div className="text-[10px] text-muted flex items-center gap-1 justify-end">
                  <Clock className="w-3 h-3" />
                  {sc.time}
                </div>
              </div>
              <Calculator className="w-3.5 h-3.5 text-muted opacity-0 group-hover:opacity-60 transition-opacity shrink-0" />
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
