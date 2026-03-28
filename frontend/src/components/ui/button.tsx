import { forwardRef } from 'react'
import { cn } from '../../lib/utils'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'ghost' | 'outline' | 'destructive' | 'success'
  size?: 'sm' | 'md' | 'lg' | 'icon'
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'md', ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center gap-2 rounded font-medium transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-accent/50 disabled:opacity-50 disabled:cursor-not-allowed select-none',
          {
            'bg-accent text-background hover:bg-accent/80 active:bg-accent/70': variant === 'default',
            'bg-transparent hover:bg-elevated text-foreground': variant === 'ghost',
            'bg-transparent border border-border text-foreground hover:bg-elevated': variant === 'outline',
            'bg-status-red text-white hover:bg-status-red/80': variant === 'destructive',
            'bg-status-green text-background hover:bg-status-green/80': variant === 'success',
          },
          {
            'h-7 px-2 text-xs': size === 'sm',
            'h-8 px-3 text-sm': size === 'md',
            'h-10 px-4 text-sm': size === 'lg',
            'h-8 w-8 p-0': size === 'icon',
          },
          className,
        )}
        {...props}
      />
    )
  },
)

Button.displayName = 'Button'
