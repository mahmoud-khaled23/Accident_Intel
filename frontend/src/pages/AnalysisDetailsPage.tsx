import { useParams } from 'react-router-dom';
import { PageHeader } from '@/components/common/PageHeader';

export function AnalysisDetailsPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div>
      <PageHeader
        title={`Analysis Report`}
        description={`Viewing accident analysis ${id ?? ''}`}
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'History', href: '/history' },
          { label: id ?? 'Details' },
        ]}
      />
      <div className="flex items-center justify-center h-64 rounded-xl border border-dashed border-[#1e2d45] text-slate-500">
        Coming soon — Analysis details will be implemented in a later step.
      </div>
    </div>
  );
}
