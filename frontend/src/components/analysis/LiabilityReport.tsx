import { ShieldAlert, AlertTriangle, Info, CheckCircle2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { LoadingState } from '@/components/common/LoadingState';
import { EvidenceCard } from './EvidenceCard';
import type { LiabilityAssessment } from '@/types/accident';
import type { LiabilityLevel } from '@/types/common';

interface LiabilityReportProps {
  report: LiabilityAssessment | null;
  loading: boolean;
  onSeek?: (timestamp: number) => void;
}

const LEVEL_CONFIG: Record<
  LiabilityLevel,
  { label: string; className: string; icon: React.ElementType }
> = {
  potential_responsibility: {
    label: 'Evidence suggests potential responsibility',
    className: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
    icon: AlertTriangle,
  },
  shared_responsibility: {
    label: 'Shared or uncertain responsibility',
    className: 'border-blue-500/30 bg-blue-500/10 text-blue-300',
    icon: Info,
  },
  insufficient_evidence: {
    label: 'Insufficient evidence to determine responsibility',
    className: 'border-slate-500/30 bg-slate-500/10 text-slate-300',
    icon: CheckCircle2,
  },
};

function LevelBadge({ level }: { level: LiabilityLevel }) {
  const cfg = LEVEL_CONFIG[level];
  const Icon = cfg.icon;
  return (
    <div className={`rounded-lg border px-4 py-3 flex items-center gap-2 text-sm font-medium ${cfg.className}`}>
      <Icon className="h-4 w-4 shrink-0" />
      {cfg.label}
    </div>
  );
}

export function LiabilityReport({ report, loading, onSeek }: LiabilityReportProps) {
  if (loading) {
    return (
      <div className="py-16">
        <LoadingState message="Running liability reasoning analysis…" />
      </div>
    );
  }
  if (!report) return null;

  return (
    <div className="space-y-4">
      <DisclaimerBanner />

      {/* Overall assessment */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-primary" />
              Overall Assessment
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <LevelBadge level={report.overallLevel} />
          <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{report.overallAssessment}</p>

          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-400">Analysis Confidence</span>
              <span className="text-slate-200">{Math.round(report.confidenceLevel * 100)}%</span>
            </div>
            <Progress value={report.confidenceLevel * 100} />
          </div>
        </CardContent>
      </Card>

      {/* Per-vehicle analysis */}
      {report.vehicleAnalyses.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Vehicle-by-Vehicle Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue={report.vehicleAnalyses[0].vehicleId}>
              <TabsList className="mb-4 flex-wrap h-auto gap-1">
                {report.vehicleAnalyses.map((va) => (
                  <TabsTrigger key={va.vehicleId} value={va.vehicleId} className="text-xs">
                    {va.vehicleLabel}
                  </TabsTrigger>
                ))}
              </TabsList>
              {report.vehicleAnalyses.map((va) => (
                <TabsContent key={va.vehicleId} value={va.vehicleId}>
                  <div className="space-y-3">
                    <LevelBadge level={va.liabilityLevel} />
                    <p className="text-sm text-slate-300">{va.summary}</p>

                    {va.contributingActions.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                          Contributing Actions
                        </p>
                        <ul className="space-y-1">
                          {va.contributingActions.map((a, i) => (
                            <li key={i} className="flex gap-2 text-sm text-red-300">
                              <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-red-400" />
                              {a}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {va.mitigatingFactors.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                          Mitigating Factors
                        </p>
                        <ul className="space-y-1">
                          {va.mitigatingFactors.map((f, i) => (
                            <li key={i} className="flex gap-2 text-sm text-slate-300">
                              <Info className="h-3.5 w-3.5 shrink-0 mt-0.5 text-blue-400" />
                              {f}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {va.evidence.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                          Evidence
                        </p>
                        <div className="space-y-2">
                          {va.evidence.map((ev) => (
                            <EvidenceCard key={ev.id} evidence={ev} onSeek={onSeek} />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          </CardContent>
        </Card>
      )}

      {/* Violations */}
      {report.violations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Potential Traffic Violations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {report.violations.map((v) => (
                <div key={v.id} className="rounded-lg border border-[#1e2d45] bg-[#0d1525] p-4">
                  <div className="flex items-start gap-2 mb-2">
                    <AlertTriangle className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-semibold text-sm text-slate-100">{v.title}</span>
                        <Badge
                          variant={v.severity === 'major' ? 'danger' : v.severity === 'moderate' ? 'warning' : 'secondary'}
                          className="text-xs"
                        >
                          {v.severity}
                        </Badge>
                        {v.lawSection && (
                          <span className="text-xs font-mono text-slate-500">{v.lawSection}</span>
                        )}
                      </div>
                      <p className="text-sm text-slate-300 mt-1">{v.description}</p>
                    </div>
                  </div>
                  {v.evidence.length > 0 && (
                    <div className="space-y-2 mt-3">
                      {v.evidence.map((ev) => (
                        <EvidenceCard key={ev.id} evidence={ev} onSeek={onSeek} />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Conflicting evidence */}
      {report.conflictingEvidence.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-400" /> Conflicting Evidence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {report.conflictingEvidence.map((ce, i) => (
                <li key={i} className="flex gap-2 text-sm text-amber-300">
                  <span className="text-amber-500 shrink-0">•</span>
                  {ce}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Missing info */}
      {report.missingInfo.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="h-4 w-4 text-slate-400" /> Missing Information
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1">
              {report.missingInfo.map((m, i) => (
                <li key={i} className="flex gap-2 text-sm text-slate-400">
                  <span className="shrink-0">—</span>
                  {m}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Limitations */}
      {report.limitations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Limitations</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {report.limitations.map((lim, i) => (
                <li key={i} className="text-sm text-slate-400 flex gap-2">
                  <span className="shrink-0 text-slate-600">•</span>
                  {lim}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <DisclaimerBanner />
    </div>
  );
}
