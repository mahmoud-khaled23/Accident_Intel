import { Play, BookOpen, Eye } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { Evidence } from '@/types/accident';

interface EvidenceCardProps {
  evidence: Evidence;
  onSeek?: (timestamp: number) => void;
}

function formatTs(s: number): string {
  return `${Math.floor(s / 60).toString().padStart(2, '0')}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

const TYPE_CONFIG = {
  video_timestamp: { icon: Play, label: 'Video', badgeVariant: 'default' as const },
  law_citation: { icon: BookOpen, label: 'Law Citation', badgeVariant: 'secondary' as const },
  observation: { icon: Eye, label: 'Observation', badgeVariant: 'outline' as const },
};

export function EvidenceCard({ evidence, onSeek }: EvidenceCardProps) {
  const config = TYPE_CONFIG[evidence.type];
  const Icon = config.icon;

  return (
    <Card className="bg-[#0d1525]">
      <CardContent className="p-4 flex items-start gap-3">
        <div className="h-8 w-8 rounded-lg bg-primary/15 flex items-center justify-center shrink-0">
          <Icon className="h-4 w-4 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Badge variant={config.badgeVariant} className="text-xs">
              {config.label}
            </Badge>
            {evidence.timestamp !== undefined && (
              <Button
                variant="link"
                size="sm"
                className="text-xs font-mono h-auto p-0 text-primary"
                onClick={() => onSeek?.(evidence.timestamp!)}
                aria-label={`Seek to ${formatTs(evidence.timestamp)}`}
              >
                @ {formatTs(evidence.timestamp)}
              </Button>
            )}
          </div>
          <p className="text-sm text-slate-300">{evidence.description}</p>
          {evidence.lawId && (
            <p className="text-xs text-slate-500 mt-1">Law ref: {evidence.lawId}</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
