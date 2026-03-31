import { useMemo, useState } from 'react'
import type { ApiPlanRow, RouteDistance, ApiRoutePlan, RiskSettings, VehicleType } from '../../types'
import { fmtCurrency, fmt } from '../../lib/utils'
import { TrendingDown, ChevronDown, ChevronUp, Calculator } from 'lucide-react'

interface CostBenefitCardProps {
  route: RouteDistance | null
  routePlan: ApiRoutePlan | null
  vehicleTypes: VehicleType[]
  riskSettings: RiskSettings
}

interface FixedCostRow {
  key: string
  name: string
  count: number
  distanceKm: number
  costPerKm: number
  fixedDispatchCost: number
  fixedCostPerVehicle: number
  totalFixedCost: number
}

interface PlanHorizonGroup {
  id: string
  horizon: string
  rows: ApiPlanRow[]
  summaryRow: ApiPlanRow
  totalVehicles: number
  vehicleSummary: string
  fixedRows: FixedCostRow[]
  fixedTotal: number
  totalCost: number
  totalWait: number
}

const HORIZON_ORDER = ['A: now', 'B: +2h', 'C: +4h', 'D: +6h']

const HORIZON_DISPLAY: Record<string, string> = {
  'A: now': 'Сейчас',
  'B: +2h': '+2ч',
  'C: +4h': '+4ч',
  'D: +6h': '+6ч',
}

function horizonLabel(h: string): string {
  return HORIZON_DISPLAY[h] ?? h
}

function FormulaBreakdown({
  group,
  riskSettings,
  vehicleMap,
}: {
  group: PlanHorizonGroup
  riskSettings: RiskSettings
  vehicleMap: Map<string, VehicleType>
}) {
  const { summaryRow, fixedRows, fixedTotal, totalWait } = group
  const waitPenaltyPerHorizon = riskSettings.idleCostPerMinute * 120
  const dispatched = group.rows.filter(r => r.vehicle_type !== 'none' && r.vehicles_count > 0)
  const underloadTotal = dispatched.reduce((s, r) => s + r.cost_underload, 0)
  const total = fixedTotal + underloadTotal + totalWait

  return (
    <div className="mt-2 border-t border-border/50 pt-4 space-y-4">
      <div className="bg-elevated rounded-lg px-3 py-2.5 text-[11px] text-muted leading-relaxed">
        <div className="font-mono text-foreground/90">J = Σ(C<sub>рейс,i</sub> × N<sub>i</sub>)</div>
        <div className="font-mono ml-3 text-foreground/90">+ U × P<sub>empty,i</sub></div>
        <div className="font-mono ml-3 text-foreground/90">+ S × P<sub>wait,horizon</sub></div>
        <div className="mt-2 pt-2 border-t border-border/50 grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
          <div><span className="text-foreground/80">Cрейс,i</span> — км × ₽/км + фикс. стоимость ТС i</div>
          <div><span className="text-foreground/80">Ni</span> — количество ТС i</div>
          <div><span className="text-foreground/80">U</span> — пустая вместимость в горизонте</div>
          <div><span className="text-foreground/80">S</span> — остаток, перенесенный дальше</div>
          <div><span className="text-foreground/80">Pempty,i</span> — штраф за пустое место ТС i, ₽/ед.</div>
          <div><span className="text-foreground/80">Pwait,horizon</span> — штраф ожидания за горизонт</div>
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
                <th className="px-3 py-2 text-right font-medium">Км</th>
                <th className="px-3 py-2 text-right font-medium">₽/км</th>
                <th className="px-3 py-2 text-right font-medium">Фикс. ₽</th>
                <th className="px-3 py-2 text-right font-medium">Cрейс / ед.</th>
                <th className="px-3 py-2 text-right font-medium">Итого</th>
              </tr>
            </thead>
            <tbody>
              {fixedRows.length > 0 ? (
                fixedRows.map(row => (
                  <tr key={row.key} className="border-b border-border/40 last:border-0">
                    <td className="px-3 py-1.5 text-foreground">{row.name}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">{row.count}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">{fmt(row.distanceKm)}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">{fmtCurrency(row.costPerKm)}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">{row.fixedDispatchCost > 0 ? fmtCurrency(row.fixedDispatchCost) : <span className="text-muted/50">—</span>}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">{fmtCurrency(row.fixedCostPerVehicle)}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-foreground font-semibold">{fmtCurrency(row.totalFixedCost)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="px-3 py-3 text-center text-muted">
                    В этом горизонте отправки нет
                  </td>
                </tr>
              )}
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
        <div className="bg-elevated rounded-lg px-3 py-2 space-y-2 text-xs">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-muted">
            <div className="flex justify-between gap-2">
              <span>Доступно к отправке</span>
              <span className="font-mono text-foreground">{fmt(summaryRow.total_available)}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span>Отправлено</span>
              <span className="font-mono text-foreground">{fmt(summaryRow.actually_shipped)}</span>
            </div>
          </div>

          {dispatched.length ? (
            <div className="space-y-1.5">
              {dispatched.map((row, idx) => {
                const v = vehicleMap.get(row.vehicle_type)
                const capPer = v?.capacity ?? 0
                const totalCap = capPer * row.vehicles_count
                const loaded = Math.max(0, totalCap - row.empty_capacity_units)
                const penalty = v?.underloadPenalty ?? riskSettings.emptyPenaltyPerUnit
                return (
                  <div key={idx} className="flex justify-between gap-2 text-muted">
                    <span>
                      <span className="text-foreground">{v?.name ?? row.vehicle_type}</span>
                      {' · '}{fmt(loaded)}/{fmt(totalCap)} ед.
                      {' · '}<span className="text-accent">{fmtCurrency(penalty)}/ед.</span>
                    </span>
                    <span className="font-mono text-foreground">
                      {fmtCurrency(row.cost_underload)}
                    </span>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-[11px] text-muted">Отправок в этом горизонте нет</div>
          )}

          <div className="border-t border-border/40 pt-1.5 text-[11px] text-muted">
            Штраф считается по пустой вместимости каждой отправленной машины и индивидуальной ставке ТС.
          </div>
        </div>
        <div className="text-right text-xs font-mono text-accent mt-1.5 pr-1">= {fmtCurrency(underloadTotal)}</div>
      </section>

      <section>
        <div className="flex items-center gap-2 mb-2">
          <span className="w-4 h-4 rounded-sm bg-status-yellow/20 text-status-yellow text-[10px] flex items-center justify-center font-bold shrink-0">3</span>
          <span className="text-[11px] text-muted font-semibold tracking-wide">ШТРАФ ЗА ОЖИДАНИЕ</span>
        </div>
        <div className="bg-elevated rounded-lg px-3 py-2 text-xs">
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted text-[11px]">
            <span>Остаток S = {fmt(summaryRow.leftover_stock)} ед.</span>
            <span>Штраф ожидания = {fmtCurrency(riskSettings.idleCostPerMinute)}/ед./мин.</span>
            <span>Длительность горизонта = 120 мин.</span>
            <span>Pwait,horizon = {fmtCurrency(waitPenaltyPerHorizon)}/ед.</span>
          </div>
        </div>
        <div className="text-right text-xs font-mono text-status-yellow mt-1.5 pr-1">= {fmtCurrency(totalWait)}</div>
      </section>

      <div className="bg-gradient-to-r from-status-green/10 to-status-green/5 border border-status-green/30 rounded-lg px-4 py-3 flex items-center justify-between">
        <span className="text-sm font-semibold text-foreground">J итого</span>
        <div className="text-right">
          <div className="text-[11px] text-muted font-mono">
            {fmtCurrency(fixedTotal)} + {fmtCurrency(underloadTotal)} + {fmtCurrency(totalWait)}
          </div>
          <div className="text-2xl font-bold font-mono text-status-green">{fmtCurrency(total)}</div>
        </div>
      </div>
    </div>
  )
}

function formatVehicleSummary(rows: ApiPlanRow[], vehicleMap: Map<string, VehicleType>) {
  const dispatchedRows = rows.filter(row => row.vehicle_type !== 'none' && row.vehicles_count > 0)
  if (dispatchedRows.length === 0) return 'Без отправки'
  return dispatchedRows
    .map(row => {
      const name = vehicleMap.get(row.vehicle_type)?.name ?? row.vehicle_type
      return `${row.vehicles_count}× ${name}`
    })
    .join(' + ')
}

export function CostBenefitCard({ route, routePlan, vehicleTypes, riskSettings }: CostBenefitCardProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const vehicleMap = useMemo(() => {
    return new Map(vehicleTypes.map(vehicle => [vehicle.id, vehicle]))
  }, [vehicleTypes])

  const planGroups = useMemo<PlanHorizonGroup[]>(() => {
    if (!routePlan || !route) return []

    const grouped = new Map<string, ApiPlanRow[]>()
    for (const row of routePlan.plan) {
      const rows = grouped.get(row.horizon)
      if (rows) {
        rows.push(row)
      } else {
        grouped.set(row.horizon, [row])
      }
    }

    return Array.from(grouped.entries())
      .map(([horizon, rows]) => {
        const dispatchedRows = rows.filter(row => row.vehicle_type !== 'none' && row.vehicles_count > 0)
        const fixedRows = dispatchedRows.map(row => {
          const vehicle = vehicleMap.get(row.vehicle_type)
          const costPerKm = vehicle?.costPerKm ?? 0
          const fixedDispatchCost = vehicle?.fixedDispatchCost ?? 0
          const fixedCostPerVehicle = route.distanceKm * costPerKm + fixedDispatchCost
          return {
            key: `${horizon}-${row.vehicle_type}`,
            name: vehicle?.name ?? row.vehicle_type,
            count: row.vehicles_count,
            distanceKm: route.distanceKm,
            costPerKm,
            fixedDispatchCost,
            fixedCostPerVehicle,
            totalFixedCost: fixedCostPerVehicle * row.vehicles_count,
          }
        })
        return {
          id: horizon,
          horizon,
          rows,
          summaryRow: rows[0],
          totalVehicles: dispatchedRows.reduce((sum, row) => sum + row.vehicles_count, 0),
          vehicleSummary: formatVehicleSummary(rows, vehicleMap),
          fixedRows,
          fixedTotal: fixedRows.reduce((sum, row) => sum + row.totalFixedCost, 0),
          totalCost: rows.reduce((s, r) => s + r.cost_total, 0),
          totalWait: rows.reduce((s, r) => s + r.cost_wait, 0),
        }
      })
      .sort((left, right) => HORIZON_ORDER.indexOf(left.horizon) - HORIZON_ORDER.indexOf(right.horizon))
  }, [route, routePlan, vehicleMap])

  if (!route) {
    return (
      <div className="bg-surface border border-border rounded-lg h-full flex items-center justify-center">
        <div className="text-center text-muted px-8">
          <TrendingDown className="w-8 h-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Выберите маршрут в таблице слева</p>
          <p className="text-xs mt-1 opacity-60">Здесь появится детализация расчета плана</p>
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
            <div className="text-sm font-semibold text-foreground">#{route.id} · {route.fromCity} → {route.toCity}</div>
            <div className="text-[11px] text-muted mt-1">Готово к отгрузке: {fmt(route.readyToShip)} ед.</div>
          </div>
        </div>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1 overflow-y-auto">
        <div>
          <div className="section-label mb-2">План развозки по маршруту</div>
          {routePlan ? (
            <div className="bg-elevated rounded-lg overflow-hidden border border-border/50">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-muted">
                    <th className="px-3 py-2 text-left font-medium">Горизонт</th>
                    <th className="px-3 py-2 text-left font-medium">ТС</th>
                    <th className="px-3 py-2 text-right font-medium">Кол-во</th>
                    <th className="px-3 py-2 text-right font-medium">Отправлено</th>
                    <th className="px-3 py-2 text-right font-medium">Остаток</th>
                    <th className="px-3 py-2 text-right font-medium">Стоимость</th>
                  </tr>
                </thead>
                <tbody>
                  {planGroups.map(group => (
                    <tr key={group.id} className="border-b border-border/40 last:border-0">
                      <td className="px-3 py-2 text-foreground">{horizonLabel(group.horizon)}</td>
                      <td className="px-3 py-2 text-muted">{group.vehicleSummary}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted">{group.totalVehicles || '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-foreground">{fmt(group.summaryRow.actually_shipped)}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted">{fmt(group.summaryRow.leftover_stock)}</td>
                      <td className="px-3 py-2 text-right font-mono text-accent">{fmtCurrency(group.totalCost)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="bg-elevated rounded-lg border border-border/50 px-4 py-8 text-center text-xs text-muted">
              План по маршруту ещё не загружен
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="section-label">Детализация расчетов</div>
          {routePlan ? (
            planGroups.map(group => {
              const isOpen = expandedId === group.id
              return (
                <div
                  key={group.id}
                  className="bg-elevated rounded-lg overflow-hidden border border-border/50"
                >
                  <button
                    onClick={() => setExpandedId(isOpen ? null : group.id)}
                    className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-border/40 transition-colors text-left"
                  >
                    <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                      group.totalVehicles === 0
                        ? 'bg-status-red'
                        : group.summaryRow.leftover_stock >= 1
                          ? 'bg-status-yellow'
                          : 'bg-status-green'
                    }`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-semibold text-foreground">{horizonLabel(group.horizon)}</span>
                        <Calculator className="w-3 h-3 text-muted opacity-50" />
                      </div>
                      <div className="text-xs text-muted truncate">{group.vehicleSummary}</div>
                    </div>
                    <div className="text-right shrink-0 mr-1">
                      <div className="text-sm font-mono font-bold text-accent">
                        {fmtCurrency(group.totalCost)}
                      </div>
                      <div className="text-[10px] text-muted">{fmt(group.summaryRow.total_available)} ед. к отправке</div>
                    </div>
                    {isOpen
                      ? <ChevronUp className="w-4 h-4 text-muted shrink-0" />
                      : <ChevronDown className="w-4 h-4 text-muted shrink-0" />
                    }
                  </button>
                  {isOpen && (
                    <div className="px-3 pb-3">
                      <FormulaBreakdown group={group} riskSettings={riskSettings} vehicleMap={vehicleMap} />
                    </div>
                  )}
                </div>
              )
            })
          ) : (
            <div className="bg-elevated rounded-lg border border-border/50 px-4 py-8 text-center text-xs text-muted">
              Детализация станет доступна после расчета плана
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
