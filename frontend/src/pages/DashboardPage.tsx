import { PageHeader } from '@/components/common/PageHeader';

export function DashboardPage() {
  return (
    <div>
      <PageHeader
        title="Dashboard"
        description="Overview of your accident investigations"
      />
      <div className="flex items-center justify-center h-64 rounded-xl border border-dashed border-[#1e2d45] text-slate-500">
        Coming soon — Dashboard content will be implemented in a later step.
      </div>
    </div>
  );
}
