import { useState } from 'react'
import type { RiskSettings } from '../../types'
import { Slider } from '../ui/slider'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
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
  const waitMin = mapWaitMinutes(local.maxWaitMinutes)

  return (
    <div className="space-y-3">
      {/* Slider 1: Economy threshold */}
      <Card>
        <CardHeader>
          <CardTitle>Экономика</CardTitle>
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
