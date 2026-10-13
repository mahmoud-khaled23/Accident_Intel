import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Menu, Bell, Sun, Moon, User, Settings, LogOut } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { isMockMode, get } from '@/services/apiClient';

interface TopbarProps {
  pageTitle: string;
  onMenuClick: () => void;
}

export function Topbar({ pageTitle, onMenuClick }: TopbarProps) {
  const navigate = useNavigate();
  const [isDark, setIsDark] = useState<boolean>(() => {
    const stored = localStorage.getItem('accident-intel-theme');
    return stored !== 'light';
  });

  // Backend health state (only used in real mode)
  const [backendStatus, setBackendStatus] = useState<'live' | 'offline' | 'checking'>('checking');

  useEffect(() => {
    if (isMockMode) return; // no health check in demo mode
    get<{ status: string }>('/api/health')
      .then((r) => setBackendStatus(r.status === 'ok' ? 'live' : 'offline'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add('dark');
      localStorage.setItem('accident-intel-theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('accident-intel-theme', 'light');
    }
  }, [isDark]);

  return (
    <header className="sticky top-0 z-30 flex h-16 shrink-0 items-center gap-4 border-b border-[#1e2d45] bg-[#0a0f1e]/95 backdrop-blur px-4 lg:px-6">
      {/* Mobile hamburger */}
      <button
        className="flex lg:hidden items-center justify-center rounded-md p-1.5 text-slate-400 hover:text-slate-100 hover:bg-[#1a2235] transition-colors"
        onClick={onMenuClick}
        aria-label="Open sidebar"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Page title */}
      <div className="flex-1 min-w-0">
        <h2 className="text-base font-semibold text-slate-100 truncate">{pageTitle}</h2>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-1">
        {/* Mode status pill */}
        {isMockMode ? (
          <span className="hidden sm:inline-flex items-center rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-400 mr-1">
            Demo
          </span>
        ) : backendStatus === 'live' ? (
          <span className="hidden sm:inline-flex items-center rounded-full border border-green-500/40 bg-green-500/10 px-2 py-0.5 text-[10px] font-medium text-green-400 mr-1">
            Live
          </span>
        ) : backendStatus === 'offline' ? (
          <span className="hidden sm:inline-flex items-center rounded-full border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-[10px] font-medium text-red-400 mr-1">
            Offline
          </span>
        ) : null}

        {/* Notification bell */}
        <Button variant="ghost" size="icon" className="relative" aria-label="Notifications">
          <Bell className="h-4 w-4" />
          {/* Badge */}
          <span className="absolute top-1.5 right-1.5 flex h-2 w-2 items-center justify-center rounded-full bg-primary text-[8px] text-white" />
        </Button>

        {/* Theme toggle */}
        <Button
          variant="ghost"
          size="icon"
          aria-label="Toggle theme"
          onClick={() => setIsDark((d) => !d)}
        >
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>

        {/* User avatar + dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className="flex items-center gap-2 rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0f1e]"
              aria-label="User menu"
            >
              <Avatar className="h-8 w-8">
                <AvatarFallback>AI</AvatarFallback>
              </Avatar>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuLabel>Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => navigate('/settings')}>
              <User className="h-4 w-4" />
              Profile
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => navigate('/settings')}>
              <Settings className="h-4 w-4" />
              Settings
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => {
                // Sign Out is not implemented in demo mode — navigate to dashboard
                navigate('/dashboard');
              }}
              className="text-danger focus:text-danger"
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
