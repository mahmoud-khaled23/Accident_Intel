import type { VehicleTrack } from '@/types/accident';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface VehicleListProps {
  tracks: VehicleTrack[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

function formatTime(s: number): string {
  return `${Math.floor(s / 60).toString().padStart(2, '0')}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

export function VehicleList({ tracks, selectedId, onSelect }: VehicleListProps) {
  return (
    <div className="space-y-3">
      {tracks.map((v) => (
        <button
          key={v.id}
          onClick={() => onSelect(v.id)}
          className={cn(
            'w-full text-left rounded-xl border p-4 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
            selectedId === v.id
              ? 'border-primary/50 bg-primary/5'
              : 'border-[#1e2d45] bg-[#131d35] hover:border-primary/30',
          )}
        >
          <div className="flex items-center gap-3 mb-3">
            {/* Color swatch */}
            <span
              className="inline-block h-3 w-3 rounded-full shrink-0"
              style={{ backgroundColor: v.color }}
            />
            <span className="font-semibold text-slate-100 text-sm">{v.label}</span>
            <Badge variant="secondary" className="ml-auto text-xs">
              {v.class}
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-slate-400 mb-3">
            <span>First seen</span>
            <span className="text-slate-200 font-mono">{formatTime(v.firstSeen)}</span>
            <span>Last seen</span>
            <span className="text-slate-200 font-mono">{formatTime(v.lastSeen)}</span>
            {v.speed !== undefined && (
              <>
                <span>Est. speed</span>
                <span className="text-slate-200">{v.speed} km/h</span>
              </>
            )}
          </div>

          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-slate-400">Tracking confidence</span>
            <span className="text-slate-200">{Math.round(v.confidence * 100)}%</span>
          </div>
          <Progress value={v.confidence * 100} className="h-1.5" />

          {v.events.length > 0 && (
            <ul className="mt-3 space-y-1">
              {v.events.map((ev, i) => (
                <li key={i} className="text-xs text-slate-400 flex gap-1.5">
                  <span className="text-slate-600">•</span>
                  {ev}
                </li>
              ))}
            </ul>
          )}
        </button>
      ))}
    </div>
  );
}
