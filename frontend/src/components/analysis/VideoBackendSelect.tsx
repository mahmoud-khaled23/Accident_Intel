import { Label } from '@/components/ui/label';
import type { PipelineHealth, VideoBackend } from '@/types/pipeline';

export function VideoBackendSelect({ value, onChange, health, disabled = false }: {
  value: VideoBackend;
  onChange: (value: VideoBackend) => void;
  health: PipelineHealth | null;
  disabled?: boolean;
}) {
  const selected = value === 'auto' ? (health?.video_vlm.gpu_available ? 'local' : 'openai') : value;
  return (
    <div className="space-y-2">
      <Label htmlFor="video-backend">Video understanding model</Label>
      <select id="video-backend" value={value} disabled={disabled}
        onChange={(event) => onChange(event.target.value as VideoBackend)}
        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
        <option value="auto">Auto — local GPU when available, otherwise API</option>
        <option value="local">Local — Qwen2-VL on the server GPU</option>
        <option value="openai">API — hosted vision model</option>
      </select>
      {health && <p className="text-xs text-slate-400">
        {selected === 'local'
          ? health.video_vlm.gpu_available ? 'CUDA GPU detected. Model weights load on first use.' : 'No CUDA GPU detected. Choose API or Auto.'
          : health.video_vlm.api_configured ? 'Sampled video frames are sent to the configured vision provider.' : 'The server needs VIDEO_VLM_MODEL and VIDEO_VLM_API_KEY for API analysis.'}
      </p>}
    </div>
  );
}
