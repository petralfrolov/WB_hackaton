import { cn } from '../../lib/utils'

interface CardProps {
  children: React.ReactNode
  className?: string
}

export function Card({ children, className }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-lg bg-surface border border-border',
        className,
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: CardProps) {
  return (
    <div className={cn('px-4 pt-4 pb-2 flex items-center justify-between', className)}>
      {children}
    </div>
  )
}

export function CardTitle({ children, className }: CardProps) {
  return (
    <h3 className={cn('section-label', className)}>{children}</h3>
  )
}

export function CardContent({ children, className }: CardProps) {
  return (
    <div className={cn('px-4 pb-4', className)}>{children}</div>
  )
}
