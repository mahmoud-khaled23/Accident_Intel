import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Plus,
  History,
  Wrench,
  BookOpen,
  Settings,
  ShieldAlert,
  X,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { label: 'New Analysis', href: '/analysis/new', icon: Plus },
  { label: 'Analysis History', href: '/history', icon: History },
  { label: 'Damage Estimator', href: '/damage', icon: Wrench },
  { label: 'Traffic Laws', href: '/laws', icon: BookOpen },
  { label: 'Settings', href: '/settings', icon: Settings },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export function Sidebar({ open, onClose }: SidebarProps) {
  const location = useLocation();

  const isActive = (href: string) => {
    if (href === '/dashboard') return location.pathname === '/dashboard';
    return location.pathname.startsWith(href);
  };

  const sidebarContent = (
    <div className="flex h-full flex-col bg-[#0d1526] border-r border-[#1e2d45]">
      {/* Logo / brand */}
      <div className="flex h-16 shrink-0 items-center gap-3 px-5 border-b border-[#1e2d45]">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20 shrink-0">
          <ShieldAlert className="h-5 w-5 text-primary" />
        </div>
        <span className="text-base font-semibold text-slate-100 tracking-tight">
          Accident-Intel
        </span>
        {/* Mobile close button */}
        <button
          className="ml-auto flex lg:hidden items-center justify-center rounded-md p-1 text-slate-400 hover:text-slate-100 hover:bg-[#1e2d45] transition-colors"
          onClick={onClose}
          aria-label="Close sidebar"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-0.5">
        {NAV_ITEMS.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.href}
              to={item.href}
              onClick={onClose}
              className={cn(
                'group flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors',
                active
                  ? 'bg-primary/15 text-primary border-l-2 border-primary pl-[10px]'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-[#1a2235] border-l-2 border-transparent pl-[10px]',
              )}
            >
              <item.icon
                className={cn(
                  'h-4 w-4 shrink-0',
                  active ? 'text-primary' : 'text-slate-500 group-hover:text-slate-300',
                )}
              />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Version tag */}
      <div className="shrink-0 border-t border-[#1e2d45] px-5 py-3">
        <span className="text-xs text-slate-600 font-mono">v0.1.0-beta</span>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop sidebar — always visible */}
      <aside className="hidden lg:flex lg:w-64 lg:shrink-0 lg:flex-col h-screen sticky top-0">
        {sidebarContent}
      </aside>

      {/* Mobile sidebar — overlay drawer */}
      {open && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40 bg-black/60 lg:hidden"
            onClick={onClose}
            aria-hidden="true"
          />
          {/* Drawer */}
          <aside className="fixed inset-y-0 left-0 z-50 w-64 flex flex-col lg:hidden">
            {sidebarContent}
          </aside>
        </>
      )}
    </>
  );
}
