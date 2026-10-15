import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Download,
  Printer,
  Copy,
  Wrench,
  ChevronLeft,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { PageHeader } from '@/components/common/PageHeader';
import { DemoBadge } from '@/components/common/DemoBadge';
import { StatusBadge } from '@/components/common/StatusBadge';
import { LoadingState } from '@/components/common/LoadingState';
import { ErrorState } from '@/components/common/ErrorState';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { VLMDescription } from '@/components/analysis/VLMDescription';
import { LegalCitationCard } from '@/components/analysis/LegalCitationCard';
import { LiabilityReport } from '@/components/analysis/LiabilityReport';
import { EvidenceCard } from '@/components/analysis/EvidenceCard';
import { VehicleList } from '@/components/tracking/VehicleList';
import { TrackingPlayer } from '@/components/tracking/TrackingPlayer';
import { getAccident, getRetrievedLaws, pollStatus, triggerAnalysis } from '@/services/accidentService';
import { apiUrl } from '@/services/apiClient';
import { exportElementToPDF } from '@/utils/pdfExport';
import type { AccidentRecord } from '@/types/accident';
import type { TrafficLaw } from '@/types/law';
import type { VideoBackend } from '@/types/pipeline';
import { usePipelineHealth } from '@/hooks/usePipelineHealth';
import { VideoBackendSelect } from '@/components/analysis/VideoBackendSelect';

// ── Main Page ──────────────────────────────────────────────────────────────

export function AnalysisDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const reportRef = useRef<HTMLDivElement>(null);
  const loadVersion = useRef(0);
  const { health } = usePipelineHealth();
  const [retryBackend, setRetryBackend] = useState<VideoBackend | null>(null);

  // Data state
  const [record, setRecord] = useState<AccidentRecord | null>(null);
  const [laws, setLaws] = useState<TrafficLaw[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Controlled tab state — enables programmatic tab switching from OverviewTab
  const [activeTab, setActiveTab] = useState('overview');

  // Tracking tab state
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | null>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(true);
  const [starting, setStarting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [seekTo, setSeekTo] = useState<number | undefined>(undefined);

  // ── Load ────────────────────────────────────────────────────────────

  const loadData = useCallback(async (quiet = false) => {
    if (!id) return;
    const version = ++loadVersion.current;
    if (!quiet) setLoading(true);
    setError(null);
    try {
      const [rec, retrievedLaws] = await Promise.all([
        getAccident(id),
        getRetrievedLaws(id),
      ]);
      if (version !== loadVersion.current) return;
      setRecord(rec);
      setLaws(retrievedLaws);
    } catch (err) {
      if (version !== loadVersion.current) return;
      const msg = err instanceof Error ? err.message : 'Failed to load analysis.';
      setError(msg.includes('not found') ? 'Analysis not found.' : msg);
    } finally {
      if (version === loadVersion.current) setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    setRetryBackend(null);
    void loadData();
    return () => { loadVersion.current += 1; };
  }, [loadData]);

  useEffect(() => {
    const currentStatus = record?.status;
    if (!id || !currentStatus || ['created', 'uploaded', 'completed', 'failed'].includes(currentStatus)) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await pollStatus(id!);
        if (cancelled) return;
        setConnectionError(null);
        if (next.status !== currentStatus) await loadData(true);
      } catch (err) {
        if (!cancelled) setConnectionError(err instanceof Error ? err.message : 'Connection lost. Reconnecting…');
      }
      if (!cancelled) timer = setTimeout(() => void poll(), 3000);
    }
    timer = setTimeout(() => void poll(), 1000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [id, record?.status, loadData]);

  async function retryAnalysis() {
    if (!id || starting) return;
    setStarting(true);
    setActionError(null);
    try {
      await triggerAnalysis(id, retryBackend ?? record?.videoBackend);
      await loadData(true);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Could not start analysis.');
    } finally { setStarting(false); }
  }

  function seekVideo(timestamp: number) {
    setSeekTo(timestamp);
    setActiveTab('tracking');
  }

  // ── Actions ──────────────────────────────────────────────────────────

  async function handleDownloadPDF() {
    if (!reportRef.current) {
      setActiveTab('report');
      setTimeout(() => void handleDownloadPDF(), 100);
      return;
    }
    await exportElementToPDF(reportRef.current, {
      filename: `accident-intel-${id ?? 'report'}.pdf`,
      onError: (msg) => alert(msg),
    });
  }

  async function handleCopySummary() {
    if (!record) return;
    const summary =
      `Accident-Intel Report — ${record.id}\n\n` +
      `Location: ${record.location ?? 'N/A'}\n` +
      `Date: ${record.incidentDate ?? 'N/A'}\n` +
      `Vehicles: ${record.vehicleCount}\n\n` +
      `Assessment: ${record.liabilityAssessment?.overallAssessment ?? 'N/A'}\n\n` +
      `Summary: ${record.vlmAnalysis?.summary ?? 'N/A'}`;
    await navigator.clipboard.writeText(summary).catch(() => undefined);
  }

  // ── Render ───────────────────────────────────────────────────────────

  if (loading) return <LoadingState message="Loading accident analysis…" />;

  if (error || !record) {
    return (
      <ErrorState
        message={error ?? 'Analysis not found.'}
        onRetry={loadData}
      />
    );
  }

  const tracks = record.tracks ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <PageHeader
        title={`Analysis Report — ${record.id.slice(0, 8)}`}
        description={record.description ?? record.videoFilename}
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'History', href: '/history' },
          { label: record.id },
        ]}
        actions={
          <div className="flex items-center gap-2 flex-wrap">
            {record.isDemoData && <DemoBadge />}
            <Button size="sm" onClick={handleDownloadPDF}>
              <Download className="h-4 w-4" />
              Download PDF
            </Button>
            <Button size="sm" variant="outline" onClick={() => window.print()}>
              <Printer className="h-4 w-4" />
              Print
            </Button>
          </div>
        }
      />

      <div className="rounded-lg border border-[#1e2d45] p-4 space-y-3" aria-live="polite">
        <StatusBadge status={record.status} />
        {record.videoModel && <p className="text-xs text-slate-400">Video model: {record.videoModel.backend} · {record.videoModel.model ?? 'Demo'}</p>}
        {!['created', 'uploaded', 'completed', 'failed'].includes(record.status) && (
          <p className="text-sm text-slate-300">{record.status === 'processing' ? 'Queued for analysis.' : 'Analysis is running.'} Results appear as each stage finishes. You can return here from History.</p>
        )}
        {record.error && <p role="alert" className="text-sm text-red-300">{record.error}</p>}
        {connectionError && <p className="text-sm text-amber-300">Connection interrupted: {connectionError}. Retrying…</p>}
        {actionError && <p role="alert" className="text-sm text-red-300">{actionError}</p>}
        {(record.status === 'uploaded' || record.status === 'failed') && (
          <div className="space-y-3 max-w-xl">
            <VideoBackendSelect value={retryBackend ?? record.videoBackend ?? 'auto'} onChange={setRetryBackend} health={health} disabled={starting} />
            <Button disabled={starting} onClick={() => void retryAnalysis()}>{starting ? 'Starting…' : record.status === 'failed' ? 'Retry analysis' : 'Start analysis'}</Button>
          </div>
        )}
        <ol className="flex flex-wrap gap-3 text-xs text-slate-400">
          {['tracking', 'vlm_analysis', 'law_retrieval', 'reasoning'].map((stage) => <li key={stage} className={stage === record.status ? 'text-primary font-semibold' : ''}>{stage.replace('_', ' ')}</li>)}
        </ol>
      </div>

      {/* Tab navigation */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="flex-wrap h-auto gap-1">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="tracking">
            Tracking
          </TabsTrigger>
          <TabsTrigger value="vlm" disabled={!record.vlmAnalysis}>
            VLM Analysis
          </TabsTrigger>
          <TabsTrigger value="laws" disabled={laws.length === 0}>
            Traffic Laws
          </TabsTrigger>
          <TabsTrigger value="liability" disabled={!record.liabilityAssessment}>
            Liability
          </TabsTrigger>
          <TabsTrigger value="report">Full Report</TabsTrigger>
        </TabsList>

        {/* ── Tab: Overview ─────────────────────────────────────────── */}
        <TabsContent value="overview" className="space-y-6">
          <OverviewTab record={record} onGoToTab={setActiveTab} />
        </TabsContent>

        {/* ── Tab: Tracking ─────────────────────────────────────────── */}
        <TabsContent value="tracking" className="space-y-6">
          {record.isDemoData && (
            <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
              <DemoBadge />
              <span>Demo Mode — Mock bounding boxes for illustration. No real tracking is running.</span>
            </div>
          )}

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Video placeholder + overlay */}
            <div className="space-y-2">
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Tracking Visualization
                </p>
                <div className="flex items-center gap-2">
                  <Label htmlFor="overlay-toggle-detail" className="text-xs text-slate-400 cursor-pointer">
                    Overlay
                  </Label>
                  <Switch
                    id="overlay-toggle-detail"
                    checked={overlayEnabled}
                    onCheckedChange={setOverlayEnabled}
                  />
                </div>
              </div>
              <TrackingPlayer src={apiUrl(`/api/accidents/${id}/video`)} tracks={tracks} selectedId={selectedVehicleId} enabled={overlayEnabled} seekTo={seekTo} />
            </div>

            {/* Vehicle list */}
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Tracked Vehicles
              </p>
              {tracks.length === 0 && <p className="text-sm text-slate-400">{record.status === 'completed' || record.vlmAnalysis ? 'No vehicles were detected in this video.' : 'Vehicle tracks will appear after tracking finishes.'}</p>}
              <VehicleList
                tracks={tracks}
                selectedId={selectedVehicleId}
                onSelect={(vid) =>
                  setSelectedVehicleId((prev) => (prev === vid ? null : vid))
                }
              />
            </div>
          </div>


        </TabsContent>

        {/* ── Tab: VLM Analysis ─────────────────────────────────────── */}
        <TabsContent value="vlm" className="space-y-4 max-w-3xl">
          {record.isDemoData && (
            <div className="flex items-center gap-2 rounded-lg border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-300">
              <DemoBadge />
              <span>VLM observations are generated from demo data in this prototype.</span>
            </div>
          )}
          <VLMDescription
            description={record.vlmAnalysis ?? null}
            loading={false}
            onSeek={seekVideo}
          />
        </TabsContent>

        {/* ── Tab: Traffic Laws ─────────────────────────────────────── */}
        <TabsContent value="laws" className="space-y-4 max-w-3xl">
          {record.isDemoData && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
              <strong>SAMPLE DATA:</strong> Law passages are demo excerpts for UI demonstration only. Do not use as legal reference.
            </div>
          )}
          <p className="text-xs text-slate-500">{laws.length} laws retrieved for this analysis</p>
          <div className="space-y-4">
            {laws.map((law) => (
              <LegalCitationCard key={law.id} law={law} />
            ))}
          </div>
        </TabsContent>

        {/* ── Tab: Liability ────────────────────────────────────────── */}
        <TabsContent value="liability" className="space-y-4 max-w-3xl">
          <LiabilityReport
            report={record.liabilityAssessment ?? null}
            loading={false}
            onSeek={seekVideo}
          />
          {record.liabilityAssessment && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                All Evidence
              </p>
              <div className="grid sm:grid-cols-2 gap-3">
                {record.liabilityAssessment.vehicleAnalyses.flatMap((va) =>
                  va.evidence.map((ev) => (
                    <EvidenceCard key={ev.id} evidence={ev} onSeek={seekVideo} />
                  )),
                )}
              </div>
            </div>
          )}
        </TabsContent>

        {/* ── Tab: Full Report ──────────────────────────────────────── */}
        <TabsContent value="report" className="space-y-6">
          <FullReportTab
            record={record}
            laws={laws}
            reportRef={reportRef}
            onDownloadPDF={handleDownloadPDF}
            onCopySummary={handleCopySummary}
            onNavigateBack={() => navigate('/history')}
            onEstimateRepair={() => navigate('/damage')}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ── Overview Tab ───────────────────────────────────────────────────────────

interface OverviewTabProps {
  record: AccidentRecord;
  onGoToTab: (tab: string) => void;
}

function OverviewTab({ record }: OverviewTabProps) {
  const liabilityLabel: Record<AccidentRecord['liabilityStatus'], string> = {
    pending: 'Pending',
    assessed: 'Assessed',
    insufficient_evidence: 'Insufficient Evidence',
  };
  const liabilityVariant: Record<
    AccidentRecord['liabilityStatus'],
    'secondary' | 'warning' | 'success'
  > = {
    pending: 'secondary',
    assessed: 'success',
    insufficient_evidence: 'warning',
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <DisclaimerBanner />

      {/* Metadata card */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Accident Metadata</CardTitle>
            {record.isDemoData && <DemoBadge className="ml-auto" />}
          </div>
        </CardHeader>
        <CardContent>
          <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-3 text-sm">
            <div>
              <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                Accident ID
              </dt>
              <dd className="text-slate-100 font-mono">{record.id}</dd>
            </div>
            <div>
              <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                Video File
              </dt>
              <dd className="text-slate-100 truncate">{record.videoFilename}</dd>
            </div>
            <div>
              <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                Upload Date
              </dt>
              <dd className="text-slate-100">
                {new Date(record.uploadDate).toLocaleString('en-US', {
                  dateStyle: 'medium',
                  timeStyle: 'short',
                })}
              </dd>
            </div>
            {record.incidentDate && (
              <div>
                <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                  Incident Date
                </dt>
                <dd className="text-slate-100">
                  {new Date(record.incidentDate).toLocaleString('en-US', {
                    dateStyle: 'medium',
                    timeStyle: 'short',
                  })}
                </dd>
              </div>
            )}
            {record.location && (
              <div className="sm:col-span-2">
                <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                  Location
                </dt>
                <dd className="text-slate-100">{record.location}</dd>
              </div>
            )}
            <div>
              <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                Vehicles
              </dt>
              <dd className="text-slate-100 font-medium">{record.vehicleCount}</dd>
            </div>
            <div>
              <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                Analysis Status
              </dt>
              <dd>
                <StatusBadge status={record.status} />
              </dd>
            </div>
            <div>
              <dt className="text-slate-400 text-xs uppercase tracking-wider mb-0.5">
                Liability Status
              </dt>
              <dd>
                <Badge variant={liabilityVariant[record.liabilityStatus]} className="text-xs">
                  {liabilityLabel[record.liabilityStatus]}
                </Badge>
              </dd>
            </div>
          </dl>

          {record.description && (
            <div className="mt-4 pt-4 border-t border-[#1e2d45]">
              <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Description</p>
              <p className="text-sm text-slate-300">{record.description}</p>
            </div>
          )}
          {record.notes && (
            <div className="mt-3">
              <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Notes</p>
              <p className="text-sm text-slate-400 italic">{record.notes}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick vehicle summary */}
      {record.tracks && record.tracks.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Vehicle Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid sm:grid-cols-2 gap-4">
              {record.tracks.map((t) => (
                <div
                  key={t.id}
                  className="rounded-lg border border-[#1e2d45] bg-[#0d1525] p-4"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className="h-3 w-3 rounded-full shrink-0"
                      style={{ backgroundColor: t.color }}
                    />
                    <span className="font-semibold text-sm text-slate-100">{t.label}</span>
                    <Badge variant="secondary" className="ml-auto text-xs">
                      {t.class}
                    </Badge>
                  </div>
                  <p className="text-xs text-slate-400">
                    Seen: {t.firstSeen.toFixed(1)}s – {t.lastSeen.toFixed(1)}s
                  </p>
                  {t.speed !== undefined && (
                    <p className="text-xs text-slate-400">Est. speed: {t.speed} km/h</p>
                  )}
                  <p className="text-xs text-slate-400">
                    Tracking confidence: {Math.round(t.confidence * 100)}%
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* VLM summary snippet */}
      {record.vlmAnalysis && (
        <Card>
          <CardHeader>
            <CardTitle>VLM Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
              {record.vlmAnalysis.summary}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Liability overall */}
      {record.liabilityAssessment && (
        <Card>
          <CardHeader>
            <CardTitle>Liability Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
              {record.liabilityAssessment.overallAssessment}
            </p>
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
              <span>Confidence:</span>
              <span className="text-slate-200 font-medium">
                {Math.round(record.liabilityAssessment.confidenceLevel * 100)}%
              </span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ── Full Report Tab ────────────────────────────────────────────────────────

interface FullReportTabProps {
  record: AccidentRecord;
  laws: TrafficLaw[];
  reportRef: React.RefObject<HTMLDivElement>;
  onDownloadPDF: () => void;
  onCopySummary: () => void;
  onNavigateBack: () => void;
  onEstimateRepair: () => void;
}

function FullReportTab({
  record,
  laws,
  reportRef,
  onDownloadPDF,
  onCopySummary,
  onNavigateBack,
  onEstimateRepair,
}: FullReportTabProps) {
  return (
    <div className="space-y-6 max-w-4xl">
      {/* Action buttons */}
      <div className="flex flex-wrap gap-2">
        <Button onClick={onDownloadPDF} size="sm">
          <Download className="h-4 w-4" />
          Download PDF
        </Button>
        <Button onClick={() => window.print()} variant="outline" size="sm">
          <Printer className="h-4 w-4" />
          Print
        </Button>
        <Button onClick={onCopySummary} variant="outline" size="sm">
          <Copy className="h-4 w-4" />
          Copy Summary
        </Button>
        <Button onClick={onNavigateBack} variant="ghost" size="sm">
          <ChevronLeft className="h-4 w-4" />
          Back to History
        </Button>
        <Button onClick={onEstimateRepair} variant="secondary" size="sm">
          <Wrench className="h-4 w-4" />
          Estimate Repair Costs
        </Button>
      </div>

      {/* Printable report body */}
      <div ref={reportRef} className="space-y-6">
        {record.isDemoData && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
            <DemoBadge />
            <span>This report is generated from mock/demo data for prototype purposes only.</span>
          </div>
        )}

        <DisclaimerBanner />

        {/* Accident Metadata */}
        <Card>
          <CardHeader>
            <CardTitle>Accident Metadata</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-2 text-sm">
              <div>
                <dt className="text-slate-400">Accident ID</dt>
                <dd className="text-slate-100 font-mono">{record.id}</dd>
              </div>
              <div>
                <dt className="text-slate-400">Video File</dt>
                <dd className="text-slate-100 truncate">{record.videoFilename}</dd>
              </div>
              <div>
                <dt className="text-slate-400">Location</dt>
                <dd className="text-slate-100">{record.location ?? 'N/A'}</dd>
              </div>
              <div>
                <dt className="text-slate-400">Incident Date</dt>
                <dd className="text-slate-100">{record.incidentDate ?? 'N/A'}</dd>
              </div>
              <div>
                <dt className="text-slate-400">Vehicles Detected</dt>
                <dd className="text-slate-100">{record.vehicleCount}</dd>
              </div>
              <div>
                <dt className="text-slate-400">Liability Status</dt>
                <dd>
                  <Badge variant="warning" className="text-xs">
                    {record.liabilityStatus}
                  </Badge>
                </dd>
              </div>
            </dl>
            {record.description && (
              <div className="mt-4">
                <dt className="text-slate-400 text-sm">Description</dt>
                <dd className="text-slate-100 text-sm mt-1">{record.description}</dd>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Vehicle Tracking Summary */}
        {record.tracks && record.tracks.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Vehicle Tracking Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid sm:grid-cols-2 gap-4">
                {record.tracks.map((t) => (
                  <div
                    key={t.id}
                    className="rounded-lg border border-[#1e2d45] bg-[#0d1525] p-4"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className="h-3 w-3 rounded-full shrink-0"
                        style={{ backgroundColor: t.color }}
                      />
                      <span className="font-semibold text-sm text-slate-100">{t.label}</span>
                      <Badge variant="secondary" className="ml-auto text-xs">
                        {t.class}
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-400">
                      First seen: {t.firstSeen.toFixed(1)}s · Last seen:{' '}
                      {t.lastSeen.toFixed(1)}s
                    </p>
                    {t.speed !== undefined && (
                      <p className="text-xs text-slate-400">Est. speed: {t.speed} km/h</p>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Accident Timeline */}
        {record.vlmAnalysis && record.vlmAnalysis.timeline.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Accident Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="space-y-3">
                {record.vlmAnalysis.timeline.map((ev, i) => (
                  <li key={i} className="flex gap-3 text-sm">
                    {ev.timestamp !== undefined && (
                      <span className="font-mono text-xs text-primary shrink-0 w-12">
                        {ev.timestamp.toFixed(1)}s
                      </span>
                    )}
                    <span
                      className={ev.type === 'fact' ? 'text-slate-200' : 'text-amber-200'}
                    >
                      {ev.content}
                    </span>
                    <Badge
                      variant={ev.type === 'fact' ? 'default' : 'warning'}
                      className="ml-auto text-xs shrink-0"
                    >
                      {ev.type}
                    </Badge>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>
        )}

        {/* VLM Description */}
        {record.vlmAnalysis && (
          <Card>
            <CardHeader>
              <CardTitle>VLM Description Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-300 mb-3">{record.vlmAnalysis.summary}</p>
              <p className="text-sm text-slate-300">{record.vlmAnalysis.collisionDescription}</p>
            </CardContent>
          </Card>
        )}

        {/* Retrieved Laws */}
        {laws.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Retrieved Traffic Laws</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {laws.map((law) => (
                  <div
                    key={law.id}
                    className="rounded-lg border border-[#1e2d45] bg-[#0d1525] p-4"
                  >
                    <div className="flex items-start gap-2">
                      <span className="font-mono text-xs text-slate-500 shrink-0">
                        {law.section}
                      </span>
                      <span className="text-sm font-medium text-slate-100">{law.title}</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1 italic">"{law.excerpt}"</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Liability Assessment */}
        {record.liabilityAssessment && (
          <LiabilityReport
            report={record.liabilityAssessment}
            loading={false}
            onSeek={undefined}
          />
        )}

        {/* Uncertainties */}
        {record.vlmAnalysis && record.vlmAnalysis.uncertainties.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Uncertainties &amp; Limitations</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {record.vlmAnalysis.uncertainties.map((u, i) => (
                  <li key={i} className="text-sm text-amber-300 flex gap-2">
                    <span className="shrink-0">•</span>
                    {u}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        <DisclaimerBanner />
      </div>
    </div>
  );
}
