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

const TOLERANCE = 0.15; // seconds — how close a frame timestamp needs to be to currentTime

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
      // Find the closest frame to currentTime
      let closest: (typeof track.frames)[0] | null = null;
      let minDelta = Infinity;
      for (const frame of track.frames) {
        const delta = Math.abs(frame.timestamp - currentTime);
        if (delta < minDelta) {
          minDelta = delta;
          closest = frame;
        }
      }

      if (!closest || minDelta > TOLERANCE) continue;

      const { x, y, width: bw, height: bh } = closest.bbox;
      const px = x * canvas.width;
      const py = y * canvas.height;
      const pw = bw * canvas.width;
      const ph = bh * canvas.height;

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
      className="absolute inset-0 pointer-events-none"
      aria-hidden="true"
    />
  );
}
