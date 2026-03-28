import { forwardRef } from 'react'
import { cn } from '../../lib/utils'

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          'h-8 w-full rounded bg-elevated border border-border px-3 text-sm text-foreground placeholder:text-muted',
          'focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          'transition-colors',
          className,
        )}
        {...props}
      />
    )
  },
)

Input.displayName = 'Input'
