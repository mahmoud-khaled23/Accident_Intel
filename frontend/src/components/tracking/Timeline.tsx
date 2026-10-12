import { useRef } from 'react';
import type { VehicleTrack } from '@/types/accident';
import { cn } from '@/lib/utils';

interface TimelineProps {
  tracks: VehicleTrack[];
  duration: number;
  currentTime: number;
  onSeek: (timestamp: number) => void;
}

export function Timeline({ tracks, duration, currentTime, onSeek }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    if (!containerRef.current || duration <= 0) return;
    const rect = containerRef.current.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onSeek(Math.max(0, Math.min(duration, ratio * duration)));
  }

  const safeMax = duration || 1;
  const cursorPct = (currentTime / safeMax) * 100;

  return (
    <div className="rounded-xl border border-[#1e2d45] bg-[#131d35] p-4">
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3">
        Timeline
      </p>

      <div
        ref={containerRef}
        className="relative h-16 rounded-lg cursor-pointer select-none bg-[#0a0f1e] overflow-hidden"
        onClick={handleClick}
        role="slider"
        aria-label="Video timeline"
        aria-valuenow={currentTime}
        aria-valuemin={0}
        aria-valuemax={duration}
      >
        {/* Vehicle presence bars */}
        {tracks.map((track, i) => {
          const left = (track.firstSeen / safeMax) * 100;
          const width = ((track.lastSeen - track.firstSeen) / safeMax) * 100;
          const top = 8 + i * 20;
          return (
            <div
              key={track.id}
              className="absolute h-4 rounded-sm opacity-70"
              style={{
                left: `${left}%`,
                width: `${Math.max(width, 0.5)}%`,
                top: top,
                backgroundColor: track.color,
              }}
              title={`${track.label}: ${track.firstSeen.toFixed(1)}s – ${track.lastSeen.toFixed(1)}s`}
            />
          );
        })}

        {/* Event markers */}
        {tracks.flatMap((track) =>
          track.events.map((ev, i) => {
            // Parse timestamp from strings like "at t=X.Xs"
            const match = ev.match(/t=(\d+\.?\d*)s/);
            if (!match) return null;
            const ts = parseFloat(match[1]);
            const left = (ts / safeMax) * 100;
            return (
              <div
                key={`${track.id}-ev-${i}`}
                className="absolute w-0.5 top-0 bottom-0 opacity-50"
                style={{ left: `${left}%`, backgroundColor: track.color }}
                title={ev}
              />
            );
          })
        )}

        {/* Current time indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white/80 z-10 pointer-events-none"
          style={{ left: `${cursorPct}%` }}
        />
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-3">
        {tracks.map((track) => (
          <div key={track.id} className="flex items-center gap-1.5">
            <span
              className={cn('inline-block h-2.5 w-2.5 rounded-sm')}
              style={{ backgroundColor: track.color }}
            />
            <span className="text-xs text-slate-400">{track.label}</span>
          </div>
        ))}
        <div className="flex items-center gap-1.5 ml-auto">
          <span className="text-xs text-slate-500 font-mono">
            {currentTime.toFixed(1)}s / {safeMax.toFixed(1)}s
          </span>
        </div>
      </div>
    </div>
  );
}
