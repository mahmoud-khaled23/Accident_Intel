import { PageHeader } from '@/components/common/PageHeader';

export function DamageEstimatorPage() {
  return (
    <div>
      <PageHeader
        title="Damage & Repair Estimator"
        description="Upload vehicle images to assess damage and estimate repair costs"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Damage Estimator' },
        ]}
      />
      <div className="flex items-center justify-center h-64 rounded-xl border border-dashed border-[#1e2d45] text-slate-500">
        Coming soon — Damage estimator workflow will be implemented in a later step.
      </div>
    </div>
  );
}
