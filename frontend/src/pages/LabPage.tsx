import { useState } from 'react'
import { RiskSettingsPanel } from '../components/lab/RiskSettings'
import { FleetManager } from '../components/lab/FleetManager'
import { RouteManager } from '../components/lab/RouteManager'
import { useSimulationContext } from '../context/SimulationContext'

type LabTab = 'fleet' | 'routes'

export function LabPage() {
  const [activeTab, setActiveTab] = useState<LabTab>('fleet')
  const { warehouses, riskSettings, setRiskSettings, routes, setRoutes } = useSimulationContext()

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Page header */}
      <div className="px-4 py-3 border-b border-border bg-surface shrink-0">
        <h1 className="text-base font-semibold text-foreground">Бизнес-правила</h1>
        <p className="text-xs text-muted mt-0.5">
          Настройка параметров оптимизатора и управление данными.
        </p>
      </div>

      {/* Three-panel layout */}
      <div className="flex flex-1 overflow-hidden gap-0 divide-x divide-border">
        {/* Left: Risk settings */}
        <div className="w-[360px] shrink-0 overflow-y-auto p-4">
          <div className="section-label mb-4">Параметры оптимизатора</div>
          <RiskSettingsPanel settings={riskSettings} onChange={setRiskSettings} />
        </div>

        {/* Right panel: tabs for Fleet / Routes */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Tab bar */}
          <div className="flex border-b border-border px-4 shrink-0 bg-surface">
            {([['fleet', 'Парк ТС по складам'], ['routes', 'Маршруты и расстояния']] as const).map(([id, label]) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={
                  `px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ` +
                  (activeTab === id
                    ? 'border-accent text-accent'
                    : 'border-transparent text-muted hover:text-foreground')
                }
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-hidden p-4">
            {activeTab === 'fleet' ? (
              <FleetManager warehouses={warehouses} />
            ) : (
              <RouteManager warehouses={warehouses} routes={routes} onChangeRoutes={setRoutes} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
