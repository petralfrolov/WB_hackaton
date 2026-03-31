import { useState, useEffect, type ChangeEvent } from 'react'
import type { RiskSettings } from '../../types'
import { Slider } from '../ui/slider'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Input } from '../ui/input'
import { Check } from 'lucide-react'
import { fmt } from '../../lib/utils'

interface RiskSettingsProps {
  settings: RiskSettings
  onChange: (settings: RiskSettings) => void
}

function mapWaitMinutes(v: number): number {
  return Math.round((v / 100) * 360)
}

export function RiskSettingsPanel({ settings, onChange }: RiskSettingsProps) {
  const [local, setLocal] = useState<RiskSettings>(settings)

  // Sync local state when backend config is loaded and updates settings prop
  useEffect(() => {
    setLocal(settings)
  }, [settings])
  const [saved, setSaved] = useState(false)
  const [selectedCat, setSelectedCat] = useState<'compact' | 'mid' | 'large'>('compact')

  const update = (key: keyof RiskSettings, v: number) => {
    setSaved(false)
    setLocal(prev => ({ ...prev, [key]: v }))
  }

  const updateCatPenalty = (cat: 'compact' | 'mid' | 'large', v: number) => {
    const keyMap: Record<typeof cat, keyof RiskSettings> = {
      compact: 'emptyPenaltyCompact',
      mid: 'emptyPenaltyMid',
      large: 'emptyPenaltyLarge',
    }
    update(keyMap[cat], v)
    // базовое поле сохраняем для совместимости (берём значение компакт)
    if (cat === 'compact') {
      update('emptyPenaltyPerUnit', v)
    }
  }

  const apply = () => {
    onChange(local)
    setSaved(true)
    setTimeout(() => setSaved(false), 2500)
  }

  const economyPct = local.economyThreshold
  const waitMin = mapWaitMinutes(local.maxWaitMinutes)

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
            <span className="text-[10px] text-muted max-w-[44%]">0% — вызываем пустые ТС</span>
            <span className="text-[10px] text-muted max-w-[44%] text-right">100% — только полная загрузка</span>
          </div>
          <p className="text-xs text-muted mt-2">
            Минимальный порог заполняемости. Алгоритм не вызовет машину, если прогнозируемая
            загрузка ниже&nbsp;
            <span className="text-accent font-mono">{fmt(economyPct)}%</span>.
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

      <Card>
        <CardHeader>
          <CardTitle>Штраф недозагрузки</CardTitle>
          <span className="text-xs font-mono font-semibold" style={{ color: '#58A6FF' }}>
            <strong>{fmt(
              selectedCat === 'compact' ? (local.emptyPenaltyCompact ?? local.emptyPenaltyPerUnit) :
              selectedCat === 'mid' ? (local.emptyPenaltyMid ?? local.emptyPenaltyPerUnit) :
              (local.emptyPenaltyLarge ?? local.emptyPenaltyPerUnit)
            )}</strong> ₽/ед.
          </span>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            {(['compact','mid','large'] as const).map(cat => (
              <Button
                key={cat}
                variant={selectedCat === cat ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCat(cat)}
              >
                {cat === 'compact' && 'Маленький'}
                {cat === 'mid' && 'Средний'}
                {cat === 'large' && 'Большой'}
              </Button>
            ))}
          </div>

          <label className="text-xs text-muted">
            Штраф за недогруженную единицу для выбранного типа ТС
          </label>
          <Input
            type="number"
            min="0"
            step="1"
            value={String(
              selectedCat === 'compact'
                ? (local.emptyPenaltyCompact ?? local.emptyPenaltyPerUnit)
                : selectedCat === 'mid'
                  ? (local.emptyPenaltyMid ?? local.emptyPenaltyPerUnit)
                  : (local.emptyPenaltyLarge ?? local.emptyPenaltyPerUnit)
            )}
            onChange={(e: ChangeEvent<HTMLInputElement>) => {
              const v = Number(e.target.value)
              updateCatPenalty(
                selectedCat,
                Number.isFinite(v) ? Math.max(0, Math.round(v)) : 0,
              )
            }}
          />

          <div className="text-[11px] text-muted space-y-1">
            <div>Штраф (compact): <span className="font-mono text-foreground">{fmt(local.emptyPenaltyCompact ?? local.emptyPenaltyPerUnit)} ₽/ед.</span></div>
            <div>Штраф (mid): <span className="font-mono text-foreground">{fmt(local.emptyPenaltyMid ?? local.emptyPenaltyPerUnit)} ₽/ед.</span></div>
            <div>Штраф (large): <span className="font-mono text-foreground">{fmt(local.emptyPenaltyLarge ?? local.emptyPenaltyPerUnit)} ₽/ед.</span></div>
          </div>
        </CardContent>
      </Card>

      {/* Slider 2: Urgency / max wait */}
      <Card>
        <CardHeader>
          <CardTitle>Срочность</CardTitle>
          <span className="text-xs font-mono font-semibold" style={{ color: '#58A6FF' }}>
            Макс. ожидание: <strong>{waitMin} мин</strong>
          </span>
        </CardHeader>
        <CardContent>
          <Slider value={local.maxWaitMinutes} onChange={v => update('maxWaitMinutes', v)} />
          <div className="flex justify-between mt-2">
            <span className="text-[10px] text-muted max-w-[44%]">0 мин — вызов немедленно</span>
            <span className="text-[10px] text-muted max-w-[44%] text-right">360 мин — ждём до 6 ч</span>
          </div>
          <p className="text-xs text-muted mt-2">
            Максимальное время ожидания товара на складе до принудительного вызова ТС.
            Текущий лимит:&nbsp;
            <span className="text-accent font-mono">{waitMin} мин</span>.
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
