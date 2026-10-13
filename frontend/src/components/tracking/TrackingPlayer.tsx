import { useEffect, useRef, useState } from 'react';
import type { VehicleTrack } from '@/types/accident';
import { TrackingOverlay } from './TrackingOverlay';
import { Timeline } from './Timeline';

interface Props {
  src: string;
  tracks: VehicleTrack[];
  selectedId: string | null;
  enabled: boolean;
  seekTo?: number;
}

export function TrackingPlayer({ src, tracks, selectedId, enabled, seekTo }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const [size, setSize] = useState({ width: 640, height: 360 });
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (video.current && seekTo !== undefined) video.current.currentTime = seekTo;
  }, [seekTo]);

  useEffect(() => { setError(false); setTime(0); setDuration(0); }, [src]);

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-xl bg-black" style={{ aspectRatio: `${size.width}/${size.height}` }}>
        <video ref={video} src={src} className="w-full h-full" controls preload="metadata"
          onLoadedMetadata={(e) => {
            const el = e.currentTarget;
            setSize({ width: el.videoWidth || 640, height: el.videoHeight || 360 });
            setDuration(Number.isFinite(el.duration) ? el.duration : 0);
            if (seekTo !== undefined) el.currentTime = seekTo;
          }}
          onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
          onError={() => setError(true)} />
        <TrackingOverlay tracks={tracks} selectedId={selectedId} currentTime={time} enabled={enabled} width={size.width} height={size.height} />
      </div>
      {error && <p role="alert" className="text-sm text-red-300">This browser cannot play the stored video. Download it or upload an H.264 MP4 or WebM version.</p>}
      <Timeline tracks={tracks} duration={duration} currentTime={time} onSeek={(t) => { if (video.current) video.current.currentTime = t; setTime(t); }} />
      <a href={src} target="_blank" rel="noreferrer" className="text-xs text-primary underline">Open original video</a>
    </div>
  );
}
