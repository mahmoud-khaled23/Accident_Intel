import { Link } from 'react-router-dom';
import { Menu, Settings } from 'lucide-react';
import { isMockMode } from '@/services/apiClient';
import { usePipelineHealth } from '@/hooks/usePipelineHealth';

interface TopbarProps { pageTitle: string; onMenuClick: () => void }

export function Topbar({ pageTitle, onMenuClick }: TopbarProps) {
  const { health, error } = usePipelineHealth();
  const demo = isMockMode || (health && Object.values(health.pipeline).includes('demo'));
  const label = demo ? 'Demo' : error ? 'Offline' : health ? 'Connected' : 'Connecting…';
  const color = demo ? 'text-amber-400 border-amber-500/40' : error ? 'text-red-400 border-red-500/40' : 'text-slate-300 border-slate-500/40';
  return (
    <header className="sticky top-0 z-30 flex h-16 shrink-0 items-center gap-4 border-b border-[#1e2d45] bg-[#0a0f1e]/95 backdrop-blur px-4 lg:px-6">
      <button className="lg:hidden rounded-md p-1.5 text-slate-400 hover:text-slate-100" onClick={onMenuClick} aria-label="Open sidebar">
        <Menu className="h-5 w-5" />
      </button>
      <h2 className="flex-1 min-w-0 text-base font-semibold text-slate-100 truncate">{pageTitle}</h2>
      <span className={`rounded-full border px-2 py-0.5 text-xs ${color}`} role="status">{label}</span>
      <Link to="/settings" aria-label="Pipeline settings" className="rounded-md p-2 text-slate-400 hover:text-slate-100">
        <Settings className="h-4 w-4" />
      </Link>
    </header>
  );
}
