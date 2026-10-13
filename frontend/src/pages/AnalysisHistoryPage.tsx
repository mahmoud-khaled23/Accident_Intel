import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  Plus,
  Trash2,
  Eye,
  ChevronLeft,
  ChevronRight,
  X,
  History,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { PageHeader } from '@/components/common/PageHeader';
import { DemoBadge } from '@/components/common/DemoBadge';
import { StatusBadge } from '@/components/common/StatusBadge';
import { EmptyState } from '@/components/common/EmptyState';
import { LoadingState } from '@/components/common/LoadingState';
import { ErrorState } from '@/components/common/ErrorState';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { getAccidents } from '@/services/accidentService';
import { deleteAccident } from '@/services/accidentService';
import { isMockMode } from '@/services/apiClient';
import type { AccidentRecord } from '@/types/accident';
import type { AnalysisStatus } from '@/types/common';

// ── Constants ──────────────────────────────────────────────────────────────

const PAGE_SIZE = 10;

const STATUS_OPTIONS: { value: AnalysisStatus | 'all'; label: string }[] = [
  { value: 'all', label: 'All Statuses' },
  { value: 'uploaded', label: 'Uploaded' },
  { value: 'tracking', label: 'Tracking' },
  { value: 'vlm_analysis', label: 'VLM Analysis' },
  { value: 'law_retrieval', label: 'Law Retrieval' },
  { value: 'reasoning', label: 'Reasoning' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
];

const LIABILITY_LABEL: Record<AccidentRecord['liabilityStatus'], string> = {
  pending: 'Pending',
  assessed: 'Assessed',
  insufficient_evidence: 'Insufficient Evidence',
};

const LIABILITY_VARIANT: Record<
  AccidentRecord['liabilityStatus'],
  'secondary' | 'warning' | 'success'
> = {
  pending: 'secondary',
  assessed: 'success',
  insufficient_evidence: 'warning',
};

// ── Main Page ──────────────────────────────────────────────────────────────

export function AnalysisHistoryPage() {
  const navigate = useNavigate();

  // Data state
  const [allRecords, setAllRecords] = useState<AccidentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<AnalysisStatus | 'all'>('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');

  // Pagination
  const [page, setPage] = useState(1);

  // Delete dialog
  const [deleteTarget, setDeleteTarget] = useState<AccidentRecord | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);

  // Toast-style feedback
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // ── Load ────────────────────────────────────────────────────────────

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch all records (history page works with full list client-side)
      const response = await getAccidents(1, 200);
      setAllRecords(response.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analysis history.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  // Auto-dismiss toast
  useEffect(() => {
    if (!toastMessage) return;
    const id = setTimeout(() => setToastMessage(null), 3000);
    return () => clearTimeout(id);
  }, [toastMessage]);

  // ── Filtering & sorting ──────────────────────────────────────────────

  const filtered = allRecords
    .filter((r) => {
      const q = searchQuery.toLowerCase().trim();
      if (q && !r.id.toLowerCase().includes(q) && !r.videoFilename.toLowerCase().includes(q)) {
        return false;
      }
      if (statusFilter !== 'all' && r.status !== statusFilter) return false;
      if (dateFrom) {
        const from = new Date(dateFrom).getTime();
        const upload = new Date(r.uploadDate).getTime();
        if (upload < from) return false;
      }
      if (dateTo) {
        const to = new Date(dateTo).getTime() + 86_400_000; // inclusive end
        const upload = new Date(r.uploadDate).getTime();
        if (upload > to) return false;
      }
      return true;
    })
    .sort((a, b) => {
      const aTime = new Date(a.uploadDate).getTime();
      const bTime = new Date(b.uploadDate).getTime();
      return sortOrder === 'newest' ? bTime - aTime : aTime - bTime;
    });

  // Reset to page 1 whenever filters change
  useEffect(() => {
    setPage(1);
  }, [searchQuery, statusFilter, dateFrom, dateTo, sortOrder]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  // ── Delete ───────────────────────────────────────────────────────────

  function confirmDelete(record: AccidentRecord) {
    setDeleteTarget(record);
    setDeleteConfirmOpen(true);
  }

  async function handleDelete() {
    if (!deleteTarget) return;
    if (!isMockMode) {
      try {
        await deleteAccident(deleteTarget.id);
      } catch (err) {
        setToastMessage(`Delete failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setDeleteTarget(null);
        setDeleteConfirmOpen(false);
        return;
      }
    }
    setAllRecords((prev) => prev.filter((r) => r.id !== deleteTarget.id));
    setToastMessage(`Analysis ${deleteTarget.id} deleted.`);
    setDeleteTarget(null);
  }

  // ── Render ───────────────────────────────────────────────────────────

  if (loading) return <LoadingState message="Loading analysis history…" />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;

  const hasFilters =
    searchQuery !== '' ||
    statusFilter !== 'all' ||
    dateFrom !== '' ||
    dateTo !== '';

  return (
    <div className="space-y-6">
      {/* Header */}
      <PageHeader
        title="Analysis History"
        description="Browse and manage your past accident investigations"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'History' },
        ]}
        actions={
          <div className="flex items-center gap-2">
            <DemoBadge />
            <Button size="sm" onClick={() => navigate('/analysis/new')}>
              <Plus className="h-4 w-4" />
              New Analysis
            </Button>
          </div>
        }
      />

      {/* Filters */}
      <div className="rounded-xl border border-[#1e2d45] bg-[#111827] p-4 space-y-3">
        {/* Row 1: Search + Status + Sort */}
        <div className="flex flex-wrap gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500 pointer-events-none" />
            <Input
              placeholder="Search by Accident ID or filename…"
              className="pl-9"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                onClick={() => setSearchQuery('')}
                aria-label="Clear search"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>

          <Select
            value={statusFilter}
            onValueChange={(v) => setStatusFilter(v as AnalysisStatus | 'all')}
          >
            <SelectTrigger className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STATUS_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value}>
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={sortOrder}
            onValueChange={(v) => setSortOrder(v as 'newest' | 'oldest')}
          >
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="newest">Newest first</SelectItem>
              <SelectItem value="oldest">Oldest first</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Row 2: Date range */}
        <div className="flex flex-wrap gap-3 items-center">
          <span className="text-xs text-slate-400 shrink-0">Date range:</span>
          <Input
            type="date"
            className="w-40"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            aria-label="From date"
          />
          <span className="text-xs text-slate-500">to</span>
          <Input
            type="date"
            className="w-40"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            aria-label="To date"
          />
          {hasFilters && (
            <button
              className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1 ml-auto"
              onClick={() => {
                setSearchQuery('');
                setStatusFilter('all');
                setDateFrom('');
                setDateTo('');
              }}
            >
              <X className="h-3 w-3" /> Clear filters
            </button>
          )}
        </div>

        <p className="text-xs text-slate-500">
          {filtered.length} result{filtered.length !== 1 ? 's' : ''}
          {allRecords.length !== filtered.length ? ` of ${allRecords.length} total` : ''}
        </p>
      </div>

      {/* Table / empty state */}
      {paginated.length === 0 ? (
        <EmptyState
          icon={<History className="h-7 w-7" />}
          title={hasFilters ? 'No analyses match your filters' : 'No analyses found'}
          description={
            hasFilters
              ? 'Try clearing filters to see all records.'
              : 'Start by uploading a video to analyze an accident.'
          }
          action={
            !hasFilters ? (
              <Button size="sm" onClick={() => navigate('/analysis/new')}>
                <Plus className="h-4 w-4" /> New Analysis
              </Button>
            ) : undefined
          }
        />
      ) : (
        <>
          <div className="rounded-xl border border-[#1e2d45] overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[#1e2d45] bg-[#111827]">
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Accident ID
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Video Filename
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Upload Date
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Vehicles
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Liability
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {paginated.map((record, idx) => (
                    <tr
                      key={record.id}
                      className={`border-b border-[#1e2d45] hover:bg-[#111827] transition-colors ${
                        idx % 2 === 0 ? 'bg-[#0d1525]' : 'bg-[#0a0f1e]'
                      }`}
                    >
                      {/* Accident ID */}
                      <td className="px-4 py-3 font-mono text-xs text-primary">
                        {record.id}
                      </td>

                      {/* Video filename */}
                      <td className="px-4 py-3 max-w-[200px]">
                        <span
                          className="text-slate-300 truncate block"
                          title={record.videoFilename}
                        >
                          {record.videoFilename}
                        </span>
                      </td>

                      {/* Upload date */}
                      <td className="px-4 py-3 text-slate-400 whitespace-nowrap">
                        {new Date(record.uploadDate).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric',
                        })}
                      </td>

                      {/* Vehicles */}
                      <td className="px-4 py-3 text-center">
                        <span className="text-slate-300 font-medium">
                          {record.vehicleCount}
                        </span>
                      </td>

                      {/* Status */}
                      <td className="px-4 py-3">
                        <StatusBadge status={record.status} />
                      </td>

                      {/* Liability */}
                      <td className="px-4 py-3">
                        <Badge
                          variant={LIABILITY_VARIANT[record.liabilityStatus]}
                          className="text-xs"
                        >
                          {LIABILITY_LABEL[record.liabilityStatus]}
                        </Badge>
                      </td>

                      {/* Actions */}
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => navigate(`/analysis/${record.id}`)}
                          >
                            <Eye className="h-3 w-3" />
                            View Report
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-xs h-7 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                            onClick={() => confirmDelete(record)}
                            aria-label={`Delete analysis ${record.id}`}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-400">
              Page {page} of {totalPages} &nbsp;·&nbsp; {filtered.length} record
              {filtered.length !== 1 ? 's' : ''}
            </p>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" />
                Prev
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                aria-label="Next page"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </>
      )}

      {/* Delete confirm dialog */}
      <ConfirmDialog
        open={deleteConfirmOpen}
        onOpenChange={setDeleteConfirmOpen}
        title={`Delete analysis ${deleteTarget?.id ?? ''}?`}
        description="This cannot be undone. The record will be removed from your local session."
        confirmLabel="Delete"
        variant="danger"
        onConfirm={() => { void handleDelete(); }}
      />

      {/* Toast notification */}
      {toastMessage && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 rounded-lg border border-green-500/30 bg-green-500/10 px-5 py-3 text-sm text-green-300 shadow-lg backdrop-blur-sm pointer-events-none">
          {toastMessage}
        </div>
      )}
    </div>
  );
}
