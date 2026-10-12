import { PageHeader } from '@/components/common/PageHeader';

export function AnalysisHistoryPage() {
  return (
    <div>
      <PageHeader
        title="Analysis History"
        description="Browse and manage your past accident investigations"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'History' },
        ]}
      />
      <div className="flex items-center justify-center h-64 rounded-xl border border-dashed border-[#1e2d45] text-slate-500">
        Coming soon — Analysis history table will be implemented in a later step.
      </div>
    </div>
  );
}
