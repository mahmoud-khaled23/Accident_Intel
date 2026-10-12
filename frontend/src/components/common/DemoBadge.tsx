import { FlaskConical } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface DemoBadgeProps {
  className?: string;
}

export function DemoBadge({ className }: DemoBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn(
        'border-amber-500/40 bg-amber-500/10 text-amber-400 text-xs',
        className,
      )}
    >
      <FlaskConical className="h-3 w-3" />
      Demo Data
    </Badge>
  );
}
