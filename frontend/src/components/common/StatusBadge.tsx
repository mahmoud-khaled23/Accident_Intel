import { Clock, Loader2, CheckCircle2, XCircle, Video, Brain, BookOpen } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import type { AnalysisStatus } from '@/types/common';

interface StatusBadgeProps {
  status: AnalysisStatus;
  className?: string;
}

const STATUS_CONFIG: Record<
  AnalysisStatus,
  { label: string; variant: 'default' | 'secondary' | 'success' | 'warning' | 'danger' | 'outline'; Icon: React.ElementType }
> = {
  created:       { label: 'Awaiting video', variant: 'secondary', Icon: Clock },
  processing:    { label: 'Queued', variant: 'default', Icon: Loader2 },
  uploaded:      { label: 'Uploaded',        variant: 'secondary', Icon: Clock },
  tracking:      { label: 'Tracking',        variant: 'default',   Icon: Video },
  vlm_analysis:  { label: 'VLM Analysis',    variant: 'default',   Icon: Brain },
  law_retrieval: { label: 'Law Retrieval',   variant: 'default',   Icon: BookOpen },
  reasoning:     { label: 'Reasoning',       variant: 'warning',   Icon: Loader2 },
  completed:     { label: 'Completed',       variant: 'success',   Icon: CheckCircle2 },
  failed:        { label: 'Failed',          variant: 'danger',    Icon: XCircle },
};

// pull in AnalysisStatus → 'pending' | 'processing' aliases too, handled via config above
const PROCESSING_STATUSES: AnalysisStatus[] = ['tracking', 'vlm_analysis', 'law_retrieval', 'reasoning'];

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const config = STATUS_CONFIG[status];
  if (!config) return null;

  const { label, variant, Icon } = config;
  const isProcessing = PROCESSING_STATUSES.includes(status);

  return (
    <Badge variant={variant} className={className}>
      <Icon className={`h-3 w-3 ${isProcessing ? 'animate-spin' : ''}`} />
      {label}
    </Badge>
  );
}
