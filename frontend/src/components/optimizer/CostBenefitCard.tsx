import { useEffect, useMemo, useState } from 'react'
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
import { Clock, TrendingDown, ChevronDown, ChevronUp, Calculator } from 'lucide-react'

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

function FormulaBreakdown({ scenario }: { scenario: CostScenario }) {
  const b = scenario.breakdown
  const fixedTotal = b.vehicles.reduce((s, v) => s + v.fixedCost * v.count, 0)
  const emptyPenaltyTotal =
    b.wEcon * b.vehicles.reduce((s, v) => s + (v.capacity - v.load) * v.count, 0) * b.pEmpty
  const delayPenaltyTotal = b.wUrg * b.itemsWaiting * b.avgWaitMinutes * b.pDelay
  const total = computeCostFromBreakdown(b)

  return (
    <div className="mt-2 border-t border-border/50 pt-4 space-y-4">
      <div className="bg-elevated rounded-lg px-3 py-2.5 text-[11px] text-muted leading-relaxed">
        <div className="font-mono text-foreground/90">J = Σ(C<sub>fixed,i</sub> × N<sub>i</sub>)</div>
        <div className="font-mono ml-3 text-foreground/90">+ W<sub>econ</sub> × Σ((Cap<sub>i</sub> - L<sub>i</sub>) × N<sub>i</sub>) × P<sub>empty</sub></div>
        <div className="font-mono ml-3 text-foreground/90">+ W<sub>urg</sub> × m × T<sub>wait</sub> × P<sub>delay</sub></div>
        <div className="mt-2 pt-2 border-t border-border/50 grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
          <div><span className="text-foreground/80">Cfixed,i</span> — фикс. стоимость ТС i</div>
          <div><span className="text-foreground/80">Ni</span> — количество ТС i</div>
          <div><span className="text-foreground/80">Capi, Li</span> — вместимость и загрузка ТС i</div>
          <div><span className="text-foreground/80">Wecon</span> — вес штрафа недозагрузки</div>
          <div><span className="text-foreground/80">m</span> — объем товаров в ожидании</div>
          <div><span className="text-foreground/80">Twait</span> — среднее время ожидания</div>
          <div><span className="text-foreground/80">Pempty</span> — цена недозагрузки за ед.</div>
          <div><span className="text-foreground/80">Pdelay</span> — цена простоя за мин.</div>
        </div>
      </div>

      <section>
        <div className="flex items-center gap-2 mb-2">
          <span className="w-4 h-4 rounded-sm bg-status-green/20 text-status-green text-[10px] flex items-center justify-center font-bold shrink-0">1</span>
          <span className="text-[11px] text-muted font-semibold tracking-wide">ПОСТОЯННЫЕ ЗАТРАТЫ</span>
        </div>
        <div className="bg-elevated rounded-lg overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border text-muted">
                <th className="px-3 py-2 text-left font-medium">ТС</th>
                <th className="px-3 py-2 text-right font-medium">Кол-во (N)</th>
                <th className="px-3 py-2 text-right font-medium">Cfixed / ед.</th>
                <th className="px-3 py-2 text-right font-medium">Итого</th>
              </tr>
            </thead>
            <tbody>
              {b.vehicles.map((v, i) => (
                <tr key={i} className="border-b border-border/40 last:border-0">
                  <td className="px-3 py-1.5 text-foreground">{v.name}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-muted">{v.count}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-muted">{fmtCurrency(v.fixedCost)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-foreground font-semibold">{fmtCurrency(v.fixedCost * v.count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="text-right text-xs font-mono text-status-green mt-1.5 pr-1">= {fmtCurrency(fixedTotal)}</div>
      </section>

      <section>
        <div className="flex items-center gap-2 mb-2">
          <span className="w-4 h-4 rounded-sm bg-accent/20 text-accent text-[10px] flex items-center justify-center font-bold shrink-0">2</span>
          <span className="text-[11px] text-muted font-semibold tracking-wide">ШТРАФ ЗА НЕДОЗАГРУЗКУ</span>
        </div>
        <div className="bg-elevated rounded-lg px-3 py-2 space-y-1 text-xs">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-muted">
            {b.vehicles.map((v, i) => (
              <div key={i} className="flex justify-between gap-2">
                <span>{v.name}: Cap={v.capacity}, L={v.load}</span>
                <span className="font-mono text-foreground">Δ={fmt((v.capacity - v.load) * v.count)}</span>
              </div>
            ))}
          </div>
          <div className="border-t border-border/40 pt-1.5 flex gap-3 text-muted text-[11px]">
            <span>Wecon = {b.wEcon.toFixed(2)}</span>
            <span>Pempty = {fmtCurrency(b.pEmpty)}/ед.</span>
          </div>
        </div>
        <div className="text-right text-xs font-mono text-accent mt-1.5 pr-1">= {fmtCurrency(emptyPenaltyTotal)}</div>
      </section>

      <section>
        <div className="flex items-center gap-2 mb-2">
          <span className="w-4 h-4 rounded-sm bg-status-yellow/20 text-status-yellow text-[10px] flex items-center justify-center font-bold shrink-0">3</span>
          <span className="text-[11px] text-muted font-semibold tracking-wide">ШТРАФ ЗА ОЖИДАНИЕ</span>
        </div>
        <div className="bg-elevated rounded-lg px-3 py-2 text-xs">
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted text-[11px]">
            <span>Wurg = {b.wUrg.toFixed(2)}</span>
            <span>m = {fmt(b.itemsWaiting)} поз.</span>
            <span>Twait = {b.avgWaitMinutes} мин.</span>
            <span>Pdelay = {fmtCurrency(b.pDelay)}/мин.</span>
          </div>
        </div>
        <div className="text-right text-xs font-mono text-status-yellow mt-1.5 pr-1">= {fmtCurrency(delayPenaltyTotal)}</div>
      </section>

      <div className="bg-gradient-to-r from-status-green/10 to-status-green/5 border border-status-green/30 rounded-lg px-4 py-3 flex items-center justify-between">
        <span className="text-sm font-semibold text-foreground">J итого</span>
        <div className="text-right">
          <div className="text-[11px] text-muted font-mono">
            {fmtCurrency(fixedTotal)} + {fmtCurrency(emptyPenaltyTotal)} + {fmtCurrency(delayPenaltyTotal)}
          </div>
          <div className="text-2xl font-bold font-mono text-status-green">{fmtCurrency(total)}</div>
        </div>
      </div>
    </div>
  )
}

const SCENARIO_COLORS = ['#3FB950', '#58A6FF', '#D29922']

export function CostBenefitCard({ recommendation, scenarios }: CostBenefitCardProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [readyNow, setReadyNow] = useState(0)

  useEffect(() => {
    if (!recommendation) {
      setReadyNow(0)
      return
    }
    setReadyNow(recommendation.forecast)
  }, [recommendation])

  const effectiveScenarios = useMemo(() => {
    return scenarios.map(sc => {
      const breakdown = {
        ...sc.breakdown,
        itemsWaiting: readyNow,
      }
      return {
        ...sc,
        breakdown,
        cost: computeCostFromBreakdown(breakdown),
      }
    })
  }, [readyNow, scenarios])

  const chartData = effectiveScenarios.map(sc => ({ ...sc }))

  if (!recommendation) {
    return (
      <div className="bg-surface border border-border rounded-lg h-full flex items-center justify-center">
        <div className="text-center text-muted px-8">
          <TrendingDown className="w-8 h-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Выберите маршрут в таблице слева</p>
          <p className="text-xs mt-1 opacity-60">Здесь появится анализ стоимости вариантов</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-surface border border-border rounded-lg flex flex-col overflow-hidden h-full">
      <div className="px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="section-label mb-0.5">Анализ стоимости</div>
            <div className="text-sm font-semibold text-foreground">{recommendation.route}</div>
          </div>
          <div className="w-[220px] shrink-0">
            <label className="text-[10px] text-muted block mb-1 text-right">
              Сейчас готово к отгрузке на маршруте
            </label>
            <input
              type="number"
              min="0"
              value={readyNow}
              onChange={e => {
                const val = Number(e.target.value)
                setReadyNow(Number.isFinite(val) ? Math.max(0, Math.round(val)) : 0)
              }}
              className="w-full h-8 rounded bg-elevated border border-border px-2 text-sm text-foreground text-right focus:outline-none focus:ring-1 focus:ring-accent"
            />
          </div>
        </div>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1 overflow-y-auto">
        <div>
          <div className="section-label mb-2">Сравнение вариантов</div>
          <ResponsiveContainer width="100%" height={140}>
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
                tickFormatter={v => `${fmt(v as number)}`}
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
              <Bar dataKey="cost" radius={[0, 3, 3, 0]} maxBarSize={26}>
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

        <div className="space-y-2">
          <div className="section-label">Расчет стоимости варианта</div>
          {effectiveScenarios.map((sc, i) => {
            const isOpen = expandedId === sc.id
            return (
              <div
                key={sc.id}
                className="bg-elevated rounded-lg overflow-hidden border border-border/50"
              >
                <button
                  onClick={() => setExpandedId(isOpen ? null : sc.id)}
                  className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-border/40 transition-colors text-left"
                >
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ background: SCENARIO_COLORS[i] ?? '#58A6FF' }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-semibold text-foreground">{sc.name}</span>
                      <Calculator className="w-3 h-3 text-muted opacity-50" />
                    </div>
                    <div className="text-xs text-muted truncate">{sc.description}</div>
                  </div>
                  <div className="text-right shrink-0 mr-1">
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
                  {isOpen
                    ? <ChevronUp className="w-4 h-4 text-muted shrink-0" />
                    : <ChevronDown className="w-4 h-4 text-muted shrink-0" />
                  }
                </button>
                {isOpen && (
                  <div className="px-3 pb-3">
                    <FormulaBreakdown scenario={sc} />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}