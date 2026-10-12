import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';

/** Maps route pathname patterns to human-readable page titles. */
const PAGE_TITLES: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/analysis/new': 'New Accident Analysis',
  '/history': 'Analysis History',
  '/damage': 'Damage & Repair Estimator',
  '/laws': 'Traffic Laws',
  '/settings': 'Settings',
};

function getPageTitle(pathname: string): string {
  // Exact match first
  if (PAGE_TITLES[pathname]) return PAGE_TITLES[pathname];
  // Prefix match for dynamic routes like /analysis/:id
  for (const [pattern, title] of Object.entries(PAGE_TITLES)) {
    if (pathname.startsWith(pattern)) return title;
  }
  return 'Accident-Intel';
}

export function AppShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const pageTitle = getPageTitle(location.pathname);

  return (
    <div className="flex min-h-screen bg-[#0a0f1e]">
      {/* Sidebar */}
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main content column */}
      <div className="flex flex-1 flex-col min-w-0">
        <Topbar
          pageTitle={pageTitle}
          onMenuClick={() => setSidebarOpen((o) => !o)}
        />
        <main className="flex-1 overflow-auto px-4 py-6 lg:px-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
