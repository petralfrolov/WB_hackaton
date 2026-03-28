import { createContext, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { RouteDistance, RiskSettings } from '../types'
import { routeDistances, defaultRiskSettings } from '../data/mockData'

interface SimulationContextValue {
  routes: RouteDistance[]
  setRoutes: (routes: RouteDistance[]) => void
  riskSettings: RiskSettings
  setRiskSettings: (settings: RiskSettings) => void
  analysisDateTime: string
  setAnalysisDateTime: (value: string) => void
}

const SimulationContext = createContext<SimulationContextValue | null>(null)

function toDateTimeLocalValue(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`
}

function normalizeToHalfHour(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value

  const minutes = date.getMinutes()
  const normalizedMinutes = minutes < 15 ? 0 : minutes < 45 ? 30 : 60

  if (normalizedMinutes === 60) {
    date.setHours(date.getHours() + 1)
    date.setMinutes(0, 0, 0)
  } else {
    date.setMinutes(normalizedMinutes, 0, 0)
  }

  return toDateTimeLocalValue(date)
}

export function SimulationProvider({ children }: { children: ReactNode }) {
  const [routes, setRoutes] = useState<RouteDistance[]>(routeDistances)
  const [riskSettings, setRiskSettings] = useState<RiskSettings>(defaultRiskSettings)
  const [analysisDateTime, setAnalysisDateTime] = useState<string>(() => normalizeToHalfHour(toDateTimeLocalValue(new Date())))

  const setAnalysisDateTimeRounded = (value: string) => {
    setAnalysisDateTime(normalizeToHalfHour(value))
  }

  const value = useMemo(
    () => ({
      routes,
      setRoutes,
      riskSettings,
      setRiskSettings,
      analysisDateTime,
      setAnalysisDateTime: setAnalysisDateTimeRounded,
    }),
    [routes, riskSettings, analysisDateTime],
  )

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  )
}

export function useSimulationContext(): SimulationContextValue {
  const ctx = useContext(SimulationContext)
  if (!ctx) {
    throw new Error('useSimulationContext must be used within SimulationProvider')
  }
  return ctx
}
