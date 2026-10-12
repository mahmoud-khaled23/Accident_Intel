import type { DamageAssessmentRecord, DamagePart } from '@/types/damage';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DemoBadge } from '@/components/common/DemoBadge';

interface DamageAssessmentProps {
  record: DamageAssessmentRecord;
}

const SEV_VARIANT: Record<DamagePart['severity'], 'secondary' | 'warning' | 'danger'> = {
  minor: 'secondary',
  moderate: 'warning',
  severe: 'danger',
};

const SEV_LABEL: Record<DamagePart['severity'], string> = {
  minor: 'Minor',
  moderate: 'Moderate',
  severe: 'Severe',
};

export function DamageAssessment({ record }: DamageAssessmentProps) {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {record.damagedParts.length} damaged part
          {record.damagedParts.length !== 1 ? 's' : ''} detected
        </p>
        <DemoBadge />
      </div>

      {/* Parts grid */}
      <div className="grid sm:grid-cols-2 gap-4">
        {record.damagedParts.map((part) => {
          const confidencePct = Math.round(part.confidence * 100);
          return (
            <Card key={part.id}>
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <CardTitle className="text-sm font-semibold text-slate-100">
                    {part.name}
                  </CardTitle>
                  <Badge variant={SEV_VARIANT[part.severity]} className="shrink-0 text-xs">
                    {SEV_LABEL[part.severity]}
                  </Badge>
                </div>
                <p className="text-xs text-slate-500">{part.location}</p>
              </CardHeader>
              <CardContent className="space-y-3">
                {/* Damage type */}
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Damage type</span>
                  <span className="text-slate-200 font-medium">{part.damageType}</span>
                </div>

                {/* Confidence bar */}
                <div>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-slate-400">AI Confidence</span>
                    <span className="text-slate-200">{confidencePct}%</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-[#0d1525] overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        confidencePct >= 80
                          ? 'bg-green-500'
                          : confidencePct >= 60
                          ? 'bg-amber-400'
                          : 'bg-slate-500'
                      }`}
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>
                </div>

                {/* Source badge */}
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Source</span>
                  <Badge
                    variant={part.source === 'ai' ? 'default' : 'secondary'}
                    className="text-xs"
                  >
                    {part.source === 'ai' ? 'AI Detected' : 'User Confirmed'}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
