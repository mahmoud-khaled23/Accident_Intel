import { useEffect, useRef, useState } from 'react';
import { Play, Pause } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface VideoPreviewProps {
  file: File;
  /** Imperative seek: when this value changes the player seeks to it */
  seekTo?: number;
  onTimeUpdate?: (currentTime: number) => void;
  /** Expose the <video> element ref to parent */
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
}

export function VideoPreview({ file, seekTo, onTimeUpdate, videoRef: externalRef }: VideoPreviewProps) {
  const internalRef = useRef<HTMLVideoElement>(null);
  const videoEl = externalRef ?? internalRef;

  const [objectUrl, setObjectUrl] = useState<string>('');
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Create/revoke object URL
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Seek when seekTo changes
  useEffect(() => {
    if (seekTo !== undefined && videoEl.current) {
      videoEl.current.currentTime = seekTo;
    }
  }, [seekTo, videoEl]);

  function togglePlay() {
    const el = videoEl.current;
    if (!el) return;
    if (el.paused) {
      el.play().catch(() => undefined);
    } else {
      el.pause();
    }
  }

  function onScrub(e: React.ChangeEvent<HTMLInputElement>) {
    const t = parseFloat(e.target.value);
    if (videoEl.current) videoEl.current.currentTime = t;
    setCurrentTime(t);
  }

  if (!objectUrl) return null;

  return (
    <div className="rounded-xl overflow-hidden border border-[#1e2d45] bg-black">
      <video
        ref={videoEl as React.RefObject<HTMLVideoElement>}
        className="w-full aspect-video object-contain"
        onTimeUpdate={(e) => {
          const t = (e.target as HTMLVideoElement).currentTime;
          setCurrentTime(t);
          onTimeUpdate?.(t);
        }}
        onDurationChange={(e) => setDuration((e.target as HTMLVideoElement).duration)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      >
        <source src={objectUrl} type={file.type || 'video/mp4'} />
        Your browser does not support the video element.
      </video>

      <div className="bg-[#0d1525] px-4 py-3 space-y-2">
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.1}
          value={currentTime}
          onChange={onScrub}
          className="w-full h-1.5 accent-primary cursor-pointer"
          aria-label="Video seek bar"
        />
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={togglePlay}
            aria-label={playing ? 'Pause' : 'Play'}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <span className="text-xs text-slate-400 font-mono">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
      </div>
    </div>
  );
}
