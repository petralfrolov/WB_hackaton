import { useState, useEffect, useRef, type ChangeEvent } from 'react'
import type { RiskSettings, Granularity } from '../../types'
import { Slider } from '../ui/slider'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Input } from '../ui/input'
import { Check } from 'lucide-react'
import { fmt } from '../../lib/utils'

const GRANULARITY_OPTIONS: { value: Granularity; label: string; desc: string }[] = [
  { value: 0.5, label: '0.5ч', desc: '12 горизонтов, макс. детализация' },
  { value: 1, label: '1ч', desc: '6 горизонтов' },
  { value: 2, label: '2ч', desc: '3 горизонта (по умолчанию)' },
]

interface RiskSettingsProps {
  settings: RiskSettings
  onChange: (settings: RiskSettings) => void
}

export function RiskSettingsPanel({ settings, onChange }: RiskSettingsProps) {
  const [local, setLocal] = useState<RiskSettings>(settings)
  // Remember last non-zero CI level so toggling back restores it
  const lastCiLevel = useRef(settings.confidenceLevel > 0 ? settings.confidenceLevel : 0.9)

  // Sync local state when backend config is loaded and updates settings prop
  useEffect(() => {
    setLocal(settings)
  }, [settings])
  const [saved, setSaved] = useState(false)

  const update = (key: keyof RiskSettings, v: number) => {
    setSaved(false)
    setLocal(prev => ({ ...prev, [key]: v }))
  }

  const apply = () => {
    onChange(local)
    setSaved(true)
    setTimeout(() => setSaved(false), 2500)
  }

  const economyPct = local.economyThreshold

  return (
    <div className="space-y-3">
      {/* Slider 1: Fill threshold */}
      <Card>
        <CardHeader>
          <CardTitle>Заполняемость</CardTitle>
          <span className="text-xs font-mono font-semibold" style={{ color: '#58A6FF' }}>
            Мин. заполняемость: <strong>{fmt(economyPct)}%</strong>
          </span>
        </CardHeader>
        <CardContent>
          <Slider value={local.economyThreshold} onChange={v => update('economyThreshold', v)} />
          <div className="flex justify-between mt-2">
            <span className="text-[10px] text-muted max-w-[44%]">0% — без ограничения загрузки</span>
            <span className="text-[10px] text-muted max-w-[44%] text-right">100% — только полная загрузка</span>
          </div>
          <p className="text-xs text-muted mt-2">
            Минимальный процент заполнения ТС. При&nbsp;
            <span className="text-accent font-mono">{fmt(economyPct)}%</span>
            &nbsp;оптимизатор не отправит машину, если погрузка ниже этого порога.
          </p>
        </CardContent>
      </Card>

      {/* Input: idle cost */}
      <Card>
        <CardHeader>
          <CardTitle>Цена простоя</CardTitle>
          <span className="text-xs font-mono font-semibold" style={{ color: '#58A6FF' }}>
            <strong>{fmt(local.idleCostPerMinute)}</strong> ₽/мин
          </span>
        </CardHeader>
        <CardContent>
          <label className="text-xs text-muted mb-1.5 block">
            Цена минуты простоя единицы товара в статусе "Готов к отправке"
          </label>
          <Input
            type="number"
            min="0"
            step="1"
            value={String(local.idleCostPerMinute)}
            onChange={(e: ChangeEvent<HTMLInputElement>) => {
              const value = Number(e.target.value)
              update('idleCostPerMinute', Number.isFinite(value) ? Math.max(0, Math.round(value)) : 0)
            }}
          />
        </CardContent>
      </Card>

      {/* Slider 3: Confidence Level */}
      <Card>
        <CardHeader>
          <CardTitle>Уверенность прогноза</CardTitle>
          <span className="text-xs font-mono font-semibold" style={{ color: '#58A6FF' }}>
            {local.confidenceLevel === 0
              ? <span className="text-muted">Отключено</span>
              : <>Доверительная вероятность: <strong>{Math.round(local.confidenceLevel * 100)}%</strong></>
            }
          </span>
        </CardHeader>
        <CardContent>
          {/* Toggle ДИ on/off */}
          <label className="flex items-center gap-2 mb-3 cursor-pointer select-none w-fit">
            <input
              type="checkbox"
              className="w-3.5 h-3.5 accent-[#58A6FF] cursor-pointer"
              checked={local.confidenceLevel > 0}
              onChange={e => {
                if (e.target.checked) {
                  update('confidenceLevel', lastCiLevel.current)
                } else {
                  if (local.confidenceLevel > 0) lastCiLevel.current = local.confidenceLevel
                  update('confidenceLevel', 0)
                }
                setSaved(false)
              }}
            />
            <span className="text-xs text-muted">Применять доверительный интервал</span>
          </label>
          <Slider
            min={0.8}
            max={0.99}
            step={0.01}
            value={local.confidenceLevel > 0 ? local.confidenceLevel : lastCiLevel.current}
            onChange={v => {
              lastCiLevel.current = v
              if (local.confidenceLevel > 0) update('confidenceLevel', v)
            }}
            disabled={local.confidenceLevel === 0}
          />
          <div className="flex justify-between mt-2">
            <span className="text-[10px] text-muted max-w-[44%]">80% — узкий ДИ</span>
            <span className="text-[10px] text-muted max-w-[44%] text-right">99% — широкий ДИ</span>
          </div>
          <p className="text-xs text-muted mt-2">
            {local.confidenceLevel === 0
              ? 'ДИ отключён — оптимизатор использует точечный прогноз без запаса под неопределённость.'
              : 'Чем выше уверенность, тем шире диапазон прогноза и больше резерв транспорта.'
            }
          </p>
        </CardContent>
      </Card>

      {/* Granularity selector */}
      <Card>
        <CardHeader>
          <CardTitle>Корреляция маршрутов</CardTitle>
          <span className="text-xs font-mono font-semibold" style={{ color: '#58A6FF' }}>
            ρ = <strong>{local.routeCorrelation.toFixed(2)}</strong>
          </span>
        </CardHeader>
        <CardContent>
          <Slider
            min={0}
            max={1}
            step={0.05}
            value={local.routeCorrelation}
            onChange={v => update('routeCorrelation', v)}
          />
          <div className="flex justify-between mt-2">
            <span className="text-[10px] text-muted max-w-[44%]">0 — независимый спрос</span>
            <span className="text-[10px] text-muted max-w-[44%] text-right">1 — полная корреляция</span>
          </div>
          <p className="text-xs text-muted mt-2">
            Оценка корреляции спроса между маршрутами. Влияет на ширину конформального интервала при агрегации по складу.
          </p>
        </CardContent>
      </Card>

      {/* Granularity selector */}
      <Card>
        <CardHeader>
          <CardTitle>Гранулярность прогноза</CardTitle>
          <span className="text-xs font-mono font-semibold" style={{ color: '#58A6FF' }}>
            Шаг: <strong>{local.granularity}ч</strong>
          </span>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            {GRANULARITY_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => {
                  setSaved(false)
                  setLocal(prev => ({ ...prev, granularity: opt.value }))
                }}
                className={`flex-1 px-3 py-2 rounded-lg border text-sm font-medium transition-colors ${
                  local.granularity === opt.value
                    ? 'bg-accent/20 border-accent text-accent'
                    : 'bg-elevated border-border text-muted hover:border-accent/50'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted mt-2">
            {GRANULARITY_OPTIONS.find(o => o.value === local.granularity)?.desc ?? ''}
          </p>
          <p className="text-[10px] text-muted mt-1">
            Деконволюция разделяет 2-часовые прогнозы ML-моделей на более мелкие интервалы.
          </p>
        </CardContent>
      </Card>

      <Button size="lg" onClick={apply} className="w-full mt-2">
        {saved ? (
          <>
            <Check className="w-4 h-4" />
            Настройки применены
          </>
        ) : (
          'Применить настройки'
        )}
      </Button>
    </div>
  )
}
