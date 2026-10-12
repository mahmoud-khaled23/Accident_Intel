import { PageHeader } from '@/components/common/PageHeader';

export function SettingsPage() {
  return (
    <div>
      <PageHeader
        title="Settings"
        description="Manage application preferences and configuration"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Settings' },
        ]}
      />
      <div className="flex items-center justify-center h-64 rounded-xl border border-dashed border-[#1e2d45] text-slate-500">
        Coming soon — Settings panel will be implemented in a later step.
      </div>
    </div>
  );
}
