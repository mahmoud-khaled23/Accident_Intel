import { PageHeader } from '@/components/common/PageHeader';

export function NewAnalysisPage() {
  return (
    <div>
      <PageHeader
        title="New Accident Analysis"
        description="Upload footage and run an AI-powered investigation"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'New Analysis' },
        ]}
      />
      <div className="flex items-center justify-center h-64 rounded-xl border border-dashed border-[#1e2d45] text-slate-500">
        Coming soon — New Analysis workflow will be implemented in a later step.
      </div>
    </div>
  );
}
