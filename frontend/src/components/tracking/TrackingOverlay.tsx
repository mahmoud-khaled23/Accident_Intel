import { useEffect, useRef } from 'react';
import type { VehicleTrack } from '@/types/accident';

interface TrackingOverlayProps {
  tracks: VehicleTrack[];
  selectedId: string | null;
  currentTime: number;
  enabled: boolean;
  /** Dimensions of the video element in pixels */
  width: number;
  height: number;
}

const TOLERANCE = 0.5; // seconds — forgiving for sampled tracking frames

export function TrackingOverlay({ tracks, selectedId, currentTime, enabled, width, height }: TrackingOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!enabled) return;

    const drawTracks = selectedId
      ? tracks.filter((t) => t.id === selectedId)
      : tracks;

    for (const track of drawTracks) {
      // Trajectories are time ordered; binary search avoids scanning the entire
      // video for every vehicle on every playback frame.
      if (currentTime < track.firstSeen || currentTime > track.lastSeen + 0.1) continue;
      let low = 0;
      let high = track.frames.length;
      while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (track.frames[mid].timestamp < currentTime) low = mid + 1;
        else high = mid;
      }
      const nearby = [track.frames[low - 1], track.frames[low]].filter(Boolean);
      const closest = nearby.reduce<(typeof track.frames)[0] | undefined>((best, frame) =>
        !best || Math.abs(frame.timestamp - currentTime) < Math.abs(best.timestamp - currentTime) ? frame : best, undefined);
      const minDelta = closest ? Math.abs(closest.timestamp - currentTime) : Infinity;

      if (!closest || minDelta > TOLERANCE) continue;

      const { x, y, width: bw, height: bh } = closest.bbox;
      const isNorm = x <= 1.05 && y <= 1.05 && bw <= 1.05 && bh <= 1.05;
      const scaleX = isNorm ? canvas.width : canvas.width / (width > 0 ? width : 1);
      const scaleY = isNorm ? canvas.height : canvas.height / (height > 0 ? height : 1);
      const px = x * scaleX;
      const py = y * scaleY;
      const pw = bw * scaleX;
      const ph = bh * scaleY;

      // Box
      ctx.strokeStyle = track.color;
      ctx.lineWidth = 2;
      ctx.strokeRect(px, py, pw, ph);

      // Semi-transparent fill
      ctx.fillStyle = track.color + '22';
      ctx.fillRect(px, py, pw, ph);

      // Label background
      const label = track.label;
      ctx.font = 'bold 11px ui-sans-serif, system-ui, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = track.color + 'cc';
      ctx.fillRect(px, py - 18, textWidth + 8, 18);

      // Label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, px + 4, py - 4);
    }
  }, [tracks, selectedId, currentTime, enabled, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 w-full h-full pointer-events-none"
      aria-hidden="true"
    />
  );
}
