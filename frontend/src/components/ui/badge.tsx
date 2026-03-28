import { cn } from '../../lib/utils'

export type BadgeVariant = 'ok' | 'warning' | 'critical' | 'called' | 'pending' | 'info'

interface BadgeProps {
  variant?: BadgeVariant
  children: React.ReactNode
  className?: string
}

const variantStyles: Record<BadgeVariant, string> = {
  ok: 'bg-status-green/15 text-status-green border border-status-green/30',
  warning: 'bg-status-yellow/15 text-status-yellow border border-status-yellow/30',
  critical: 'bg-status-red/15 text-status-red border border-status-red/30',
  called: 'bg-status-green/15 text-status-green border border-status-green/30',
  pending: 'bg-muted/15 text-muted border border-muted/30',
  info: 'bg-accent/15 text-accent border border-accent/30',
}

export function Badge({ variant = 'info', children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide whitespace-nowrap',
        variantStyles[variant],
        className,
      )}
    >
      {children}
    </span>
  )
}
