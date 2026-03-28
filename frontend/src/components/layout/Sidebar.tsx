import { NavLink } from 'react-router-dom'
import { Globe, BarChart2, FlaskConical } from 'lucide-react'
import { cn } from '../../lib/utils'
import { useSimulationContext } from '../../context/SimulationContext'

interface NavItem {
  to: string
  icon: React.ElementType
  label: string
}

const navItems: NavItem[] = [
  { to: '/map', icon: Globe, label: 'Карта складов' },
  { to: '/optimizer', icon: BarChart2, label: 'Оптимизатор' },
  { to: '/lab', icon: FlaskConical, label: 'Лаборатория правил' },
]

export function Sidebar() {
  const { analysisDateTime, setAnalysisDateTime } = useSimulationContext()

  return (
    <aside
      className={cn(
        'relative flex flex-col h-full bg-surface border-r border-border shrink-0 z-20 w-[200px]',
      )}
    >
      {/* Logo */}
      <div className="flex items-center h-14 px-3.5 border-b border-border shrink-0">
        <div className="w-7 h-7 rounded bg-accent flex items-center justify-center shrink-0">
          <span className="text-background font-bold text-xs">WB</span>
        </div>
        <span className="ml-2.5 text-sm font-semibold text-foreground whitespace-nowrap">
          Transport DS
        </span>
      </div>

      {/* Nav items */}
      <nav className="flex flex-col gap-0.5 p-2 flex-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'relative flex items-center gap-3 h-10 px-2.5 rounded transition-colors overflow-hidden',
                isActive
                  ? 'bg-accent/10 text-accent'
                  : 'text-muted hover:bg-elevated hover:text-foreground',
              )
            }
          >
            {({ isActive }) => (
              <>
                {/* Active indicator */}
                {isActive && (
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-accent rounded-r" />
                )}
                <Icon className="w-5 h-5 shrink-0" />
                <span className="text-sm whitespace-nowrap overflow-hidden">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="p-2 border-t border-border space-y-1.5">
        <label className="text-[10px] text-muted px-1 block">Дата и время анализа</label>
        <input
          type="datetime-local"
          step={1800}
          value={analysisDateTime}
          onChange={e => setAnalysisDateTime(e.target.value)}
          className="w-full h-8 rounded bg-elevated border border-border px-2 text-[11px] text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
        />
      </div>

      {/* Version badge */}
      <div className="p-2 border-t border-border">
        <span className="text-[10px] text-muted px-2">v0.1.0 · beta</span>
      </div>
    </aside>
  )
}
