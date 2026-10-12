import { PageHeader } from '@/components/common/PageHeader';

export function TrafficLawsPage() {
  return (
    <div>
      <PageHeader
        title="NYC Traffic Law Library"
        description="Browse and search New York City traffic regulations"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Traffic Laws' },
        ]}
      />
      <div className="flex items-center justify-center h-64 rounded-xl border border-dashed border-[#1e2d45] text-slate-500">
        Coming soon — Traffic laws library will be implemented in a later step.
      </div>
    </div>
  );
}
