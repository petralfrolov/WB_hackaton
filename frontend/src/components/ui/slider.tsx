import * as RadixSlider from '@radix-ui/react-slider'
import { cn } from '../../lib/utils'

interface SliderProps {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  className?: string
  disabled?: boolean
}

export function Slider({ value, onChange, min = 0, max = 100, step = 1, className, disabled }: SliderProps) {
  return (
    <RadixSlider.Root
      className={cn('relative flex items-center select-none touch-none w-full h-5', disabled && 'opacity-40 pointer-events-none', className)}
      value={[value]}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={([v]) => onChange(v)}
    >
      <RadixSlider.Track className="relative grow rounded-full h-1 bg-elevated">
        <RadixSlider.Range className="absolute bg-accent rounded-full h-full" />
      </RadixSlider.Track>
      <RadixSlider.Thumb
        className="block w-4 h-4 bg-accent rounded-full shadow-md focus:outline-none focus:ring-2 focus:ring-accent/60 hover:bg-accent/80 transition-colors cursor-pointer"
        aria-label="Value"
      />
    </RadixSlider.Root>
  )
}
