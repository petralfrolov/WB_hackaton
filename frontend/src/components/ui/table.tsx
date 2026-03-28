import { cn } from '../../lib/utils'

export function Table({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className="w-full overflow-x-auto">
      <table className={cn('w-full caption-bottom text-sm', className)}>{children}</table>
    </div>
  )
}

export function TableHeader({ children }: { children: React.ReactNode }) {
  return <thead>{children}</thead>
}

export function TableBody({ children }: { children: React.ReactNode }) {
  return <tbody>{children}</tbody>
}

export function TableRow({
  children,
  className,
  onClick,
  selected,
}: {
  children: React.ReactNode
  className?: string
  onClick?: () => void
  selected?: boolean
}) {
  return (
    <tr
      onClick={onClick}
      className={cn(
        'border-b border-border transition-colors',
        onClick && 'cursor-pointer hover:bg-elevated',
        selected && 'bg-accent/10 hover:bg-accent/10',
        className,
      )}
    >
      {children}
    </tr>
  )
}

export function TableHead({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <th
      className={cn(
        'h-9 px-3 text-left text-[11px] font-semibold uppercase tracking-widest text-muted whitespace-nowrap',
        className,
      )}
    >
      {children}
    </th>
  )
}

export function TableCell({
  children,
  className,
  colSpan,
}: {
  children: React.ReactNode
  className?: string
  colSpan?: number
}) {
  return (
    <td colSpan={colSpan} className={cn('px-3 py-2.5 text-sm text-foreground align-middle', className)}>
      {children}
    </td>
  )
}
