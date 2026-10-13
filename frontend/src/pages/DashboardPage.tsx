import { useState, useMemo, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Car,
  FileText,
  Crosshair,
  Wrench,
  Plus,
  ChevronUp,
  ChevronDown,
  ChevronsUpDown,
  ExternalLink,
  CheckCircle2,
  UploadCloud,
  Video,
  Brain,
  BookOpen,
  Cpu,
  AlertTriangle,
  Clock,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

import { PageHeader } from '@/components/common/PageHeader';
import { DemoBadge } from '@/components/common/DemoBadge';
import { StatusBadge } from '@/components/common/StatusBadge';
import { EmptyState } from '@/components/common/EmptyState';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

import { MOCK_ACCIDENTS } from '@/mocks/accidentMocks';
import { MOCK_DAMAGE_ASSESSMENTS } from '@/mocks/damageMocks';
import { isMockMode } from '@/services/apiClient';
import { getAccidents } from '@/services/accidentService';
import type { AccidentRecord } from '@/types/accident';
import type { AnalysisStatus } from '@/types/common';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// ---------------------------------------------------------------------------
// Status chart colors
// ---------------------------------------------------------------------------
const STATUS_COLORS: Record<AnalysisStatus, string> = {
  uploaded: '#6b7280',
  tracking: '#06b6d4',
  vlm_analysis: '#8b5cf6',
  law_retrieval: '#f59e0b',
  reasoning: '#f97316',
  completed: '#22c55e',
  failed: '#ef4444',
};

// ---------------------------------------------------------------------------
// Recent activity — derived from mock accidents
// ---------------------------------------------------------------------------
interface ActivityEvent {
  id: string;
  icon: React.ElementType;
  iconColor: string;
  description: string;
  relativeTime: string;
}

function buildActivityEvents(accidents: AccidentRecord[]): ActivityEvent[] {
  const events: ActivityEvent[] = [];
  const sorted = [...accidents].sort(
    (a, b) => new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime(),
  );

  for (const acc of sorted.slice(0, 8)) {
    const iconMap: Record<AnalysisStatus, React.ElementType> = {
      uploaded: UploadCloud,
      tracking: Video,
      vlm_analysis: Brain,
      law_retrieval: BookOpen,
      reasoning: Cpu,
      completed: CheckCircle2,
      failed: AlertTriangle,
    };
    const colorMap: Record<AnalysisStatus, string> = {
      uploaded: 'text-slate-400',
      tracking: 'text-cyan-400',
      vlm_analysis: 'text-violet-400',
      law_retrieval: 'text-amber-400',
      reasoning: 'text-orange-400',
      completed: 'text-green-400',
      failed: 'text-red-400',
    };
    const labelMap: Record<AnalysisStatus, string> = {
      uploaded: 'Video uploaded',
      tracking: 'Vehicle tracking in progress',
      vlm_analysis: 'VLM analysis running',
      law_retrieval: 'Retrieving traffic laws',
      reasoning: 'Reasoning & liability analysis',
      completed: 'Analysis completed',
      failed: 'Analysis failed',
    };

    events.push({
      id: acc.id,
      icon: iconMap[acc.status],
      iconColor: colorMap[acc.status],
      description: `${labelMap[acc.status]} — ${acc.id} (${acc.videoFilename})`,
      relativeTime: timeAgo(acc.uploadDate),
    });
  }
  return events;
}

// ---------------------------------------------------------------------------
// Sortable column types
// ---------------------------------------------------------------------------
type SortKey = 'uploadDate' | 'vehicleCount' | 'id';
type SortDir = 'asc' | 'desc';

// ---------------------------------------------------------------------------
// Liability label
// ---------------------------------------------------------------------------
function LiabilityLabel({
  status,
}: {
  status: AccidentRecord['liabilityStatus'];
}) {
  if (status === 'assessed')
    return (
      <Badge variant="success">
        <CheckCircle2 className="h-3 w-3" />
        Assessed
      </Badge>
    );
  if (status === 'insufficient_evidence')
    return (
      <Badge variant="warning">
        <AlertTriangle className="h-3 w-3" />
        Insufficient
      </Badge>
    );
  return (
    <Badge variant="secondary">
      <Clock className="h-3 w-3" />
      Pending
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// DashboardPage
// ---------------------------------------------------------------------------
const PAGE_SIZE = 5;

export function DashboardPage() {
  const navigate = useNavigate();

  // ── Data: use real API in live mode, mock data in demo mode ───────────
  const [accidents, setAccidents] = useState<AccidentRecord[]>(
    isMockMode ? MOCK_ACCIDENTS : [],
  );

  useEffect(() => {
    if (isMockMode) return; // mock data already set above
    getAccidents(1, 100)
      .then((res) => setAccidents(res.items))
      .catch(() => setAccidents([]));
  }, []);

  // ── Derived stats ──────────────────────────────────────────────────────
  const totalAnalyses = isMockMode ? MOCK_ACCIDENTS.length : accidents.length;
  const reportsGenerated = accidents.filter(
    (a) => a.status === 'completed',
  ).length;
  const vehiclesTracked = accidents.reduce(
    (sum, a) => sum + a.vehicleCount,
    0,
  );
  const damageAssessments = MOCK_DAMAGE_ASSESSMENTS.length;

  // ── Table state ────────────────────────────────────────────────────────
  const [sortKey, setSortKey] = useState<SortKey>('uploadDate');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [page, setPage] = useState(1);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
    setPage(1);
  }

  const sortedAccidents = useMemo(() => {
    const source = isMockMode ? MOCK_ACCIDENTS : accidents;
    const copy = [...source];
    copy.sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'uploadDate') {
        cmp =
          new Date(a.uploadDate).getTime() - new Date(b.uploadDate).getTime();
      } else if (sortKey === 'vehicleCount') {
        cmp = a.vehicleCount - b.vehicleCount;
      } else {
        cmp = a.id.localeCompare(b.id);
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return copy;
  }, [sortKey, sortDir, accidents]);

  const totalPages = Math.ceil(sortedAccidents.length / PAGE_SIZE);
  const pageAccidents = sortedAccidents.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE,
  );

  // ── Chart data ─────────────────────────────────────────────────────────
  const chartData = useMemo(() => {
    const source = isMockMode ? MOCK_ACCIDENTS : accidents;
    const counts: Partial<Record<AnalysisStatus, number>> = {};
    for (const a of source) {
      counts[a.status] = (counts[a.status] ?? 0) + 1;
    }
    return (Object.entries(counts) as [AnalysisStatus, number][]).map(
      ([status, count]) => ({
        name: status
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (c) => c.toUpperCase()),
        value: count,
        color: STATUS_COLORS[status],
      }),
    );
  }, []);

  // ── Activity timeline ──────────────────────────────────────────────────
  const activityEvents = useMemo(
    () => buildActivityEvents(isMockMode ? MOCK_ACCIDENTS : accidents),
    [accidents],
  );

  // ── Sort icon helper ───────────────────────────────────────────────────
  function SortIcon({ col }: { col: SortKey }) {
    if (col !== sortKey) return <ChevronsUpDown className="h-3.5 w-3.5 text-slate-500" />;
    return sortDir === 'asc' ? (
      <ChevronUp className="h-3.5 w-3.5 text-primary" />
    ) : (
      <ChevronDown className="h-3.5 w-3.5 text-primary" />
    );
  }

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="space-y-6">
      {/* Page header */}
      <PageHeader
        title="Dashboard"
        description="Overview of your accident investigations"
        actions={<DemoBadge />}
      />

      {/* ── Stats row ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          icon={<Car className="h-5 w-5 text-primary" />}
          label="Total Accidents"
          value={totalAnalyses}
          trend="+2 this week"
          trendUp
        />
        <StatCard
          icon={<FileText className="h-5 w-5 text-green-400" />}
          label="Reports Generated"
          value={reportsGenerated}
          trend={`${reportsGenerated} completed`}
          trendUp
        />
        <StatCard
          icon={<Crosshair className="h-5 w-5 text-cyan-400" />}
          label="Vehicles Tracked"
          value={vehiclesTracked}
          trend="across all incidents"
        />
        <StatCard
          icon={<Wrench className="h-5 w-5 text-violet-400" />}
          label="Damage Assessments"
          value={damageAssessments}
          trend={`${damageAssessments} sessions`}
          trendUp
        />
      </div>

      {/* ── CTA buttons row ───────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-3">
        <Button size="lg" onClick={() => navigate('/analysis/new')}>
          <Plus className="h-5 w-5" />
          New Accident Analysis
        </Button>
        <Button
          size="lg"
          variant="secondary"
          onClick={() => navigate('/damage')}
        >
          <Wrench className="h-5 w-5" />
          Estimate Repair Cost
        </Button>
      </div>

      {/* ── Recent Analyses table ──────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Recent Analyses</CardTitle>
            <DemoBadge />
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {sortedAccidents.length === 0 ? (
            <EmptyState
              icon={<Car className="h-7 w-7" />}
              title="No analyses yet"
              description="Upload an accident video to get started."
              action={
                <Button onClick={() => navigate('/analysis/new')}>
                  <Plus className="h-4 w-4" />
                  New Analysis
                </Button>
              }
            />
          ) : (
            <>
              {/* Scrollable table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <Th>
                        <button
                          className="flex items-center gap-1 hover:text-slate-200 transition-colors"
                          onClick={() => handleSort('id')}
                        >
                          Accident ID
                          <SortIcon col="id" />
                        </button>
                      </Th>
                      <Th>Video Filename</Th>
                      <Th>
                        <button
                          className="flex items-center gap-1 hover:text-slate-200 transition-colors"
                          onClick={() => handleSort('uploadDate')}
                        >
                          Upload Date
                          <SortIcon col="uploadDate" />
                        </button>
                      </Th>
                      <Th>
                        <button
                          className="flex items-center gap-1 hover:text-slate-200 transition-colors"
                          onClick={() => handleSort('vehicleCount')}
                        >
                          Vehicles
                          <SortIcon col="vehicleCount" />
                        </button>
                      </Th>
                      <Th>Status</Th>
                      <Th>Liability</Th>
                      <Th>Actions</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {pageAccidents.map((acc) => (
                      <tr
                        key={acc.id}
                        className="border-b border-border/50 hover:bg-[#0f1929]/60 transition-colors"
                      >
                        <Td>
                          <span className="font-mono text-xs text-primary">
                            {acc.id}
                          </span>
                        </Td>
                        <Td>
                          <span
                            className="max-w-[180px] truncate block text-slate-300"
                            title={acc.videoFilename}
                          >
                            {acc.videoFilename}
                          </span>
                        </Td>
                        <Td>
                          <span className="text-slate-400">
                            {formatDate(acc.uploadDate)}
                          </span>
                        </Td>
                        <Td>
                          <span className="text-slate-300">
                            {acc.vehicleCount}
                          </span>
                        </Td>
                        <Td>
                          <StatusBadge status={acc.status} />
                        </Td>
                        <Td>
                          <LiabilityLabel status={acc.liabilityStatus} />
                        </Td>
                        <Td>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => navigate(`/analysis/${acc.id}`)}
                            className="gap-1"
                          >
                            <ExternalLink className="h-3.5 w-3.5" />
                            View Report
                          </Button>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-between px-5 py-3 text-sm text-slate-400">
                <span>
                  Showing {(page - 1) * PAGE_SIZE + 1}–
                  {Math.min(page * PAGE_SIZE, sortedAccidents.length)} of{' '}
                  {sortedAccidents.length}
                </span>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={page === 1}
                    onClick={() => setPage((p) => p - 1)}
                  >
                    Previous
                  </Button>
                  <span className="px-1">
                    {page} / {totalPages}
                  </span>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={page === totalPages}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* ── Bottom row: chart + activity ──────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Status Distribution Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Analysis Status Distribution</CardTitle>
              <DemoBadge />
            </div>
          </CardHeader>
          <CardContent>
            <div style={{ height: 280 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={110}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: '#1a2235',
                      border: '1px solid #1e2d45',
                      borderRadius: '8px',
                      color: '#e2e8f0',
                      fontSize: '12px',
                    }}
                    formatter={(value) => [value as number, 'Analyses']}
                  />
                  <Legend
                    formatter={(value) => (
                      <span style={{ color: '#94a3b8', fontSize: '12px' }}>
                        {value}
                      </span>
                    )}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Recent Activity</CardTitle>
              <DemoBadge />
            </div>
          </CardHeader>
          <CardContent>
            <ol className="space-y-3">
              {activityEvents.map((ev) => {
                const Icon = ev.icon;
                return (
                  <li key={ev.id} className="flex items-start gap-3">
                    <span
                      className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-background-card border border-border ${ev.iconColor}`}
                    >
                      <Icon className="h-3.5 w-3.5" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-slate-300 leading-snug line-clamp-2">
                        {ev.description}
                      </p>
                      <p className="mt-0.5 text-xs text-slate-500">
                        {ev.relativeTime}
                      </p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: number;
  trend: string;
  trendUp?: boolean;
}

function StatCard({ icon, label, value, trend, trendUp }: StatCardProps) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">
              {label}
            </p>
            <p className="text-3xl font-bold text-slate-100">{value}</p>
            <p
              className={`mt-1 text-xs ${
                trendUp === undefined
                  ? 'text-slate-500'
                  : trendUp
                  ? 'text-green-400'
                  : 'text-red-400'
              }`}
            >
              {trend}
            </p>
          </div>
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-background-card border border-border">
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-5 py-3 text-xs font-medium text-slate-400 uppercase tracking-wide whitespace-nowrap">
      {children}
    </th>
  );
}

function Td({ children }: { children: React.ReactNode }) {
  return <td className="px-5 py-3">{children}</td>;
}
