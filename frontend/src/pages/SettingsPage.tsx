import { useState } from 'react';
import { Link } from 'react-router-dom';
import { PageHeader } from '@/components/common/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { VideoBackendSelect } from '@/components/analysis/VideoBackendSelect';
import { usePipelineHealth } from '@/hooks/usePipelineHealth';
import { getSettings, saveSettings } from '@/services/settingsService';

export function SettingsPage() {
  const { health, error } = usePipelineHealth();
  const [backend, setBackend] = useState(() => getSettings().videoBackend);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  return (
    <div className="max-w-3xl space-y-6">
      <PageHeader title="Pipeline settings" description="Choose how videos are analyzed and inspect the connected server." />
      <Card>
        <CardHeader><CardTitle>Default for new uploads</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <VideoBackendSelect value={backend} onChange={(value) => { setBackend(value); setSaved(false); }} health={health} />
          <Button onClick={() => {
            try { saveSettings({ videoBackend: backend }); setSaved(true); setSaveError(null); }
            catch { setSaveError('Browser preferences could not be saved. Choose a model when uploading.'); }
          }}>Save preference</Button>
          {saved && <p role="status" className="text-sm text-green-400">Saved for new uploads in this browser.</p>}
          {saveError && <p role="alert" className="text-sm text-red-300">{saveError}</p>}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Connected pipeline</CardTitle></CardHeader>
        <CardContent className="space-y-4 text-sm">
          {error && <p role="alert" className="text-red-300">{error}</p>}
          {!health && !error && <p>Checking server…</p>}
          {health && <>
            <ol className="grid gap-3 sm:grid-cols-2">
              <li>1. Vehicle tracking · {health.pipeline.tracking}</li>
              <li>2. Video understanding · {health.pipeline.vlm}</li>
              <li>3. Traffic-law retrieval · {health.pipeline.rag}</li>
              <li>4. Evidence reasoning · {health.pipeline.reasoning}</li>
            </ol>
            <dl className="grid grid-cols-2 gap-2 text-slate-300">
              <dt>Server VLM default</dt><dd>{health.video_vlm.requested} → {health.video_vlm.backend}</dd>
              <dt>Model</dt><dd className="break-all">{health.video_vlm.model || 'Not configured'}</dd>
              <dt>CUDA GPU</dt><dd>{health.video_vlm.gpu_available ? 'Available' : 'Unavailable'}</dd>
              <dt>Vision API credentials</dt><dd>{health.video_vlm.api_configured ? 'Configured' : 'Missing'}</dd>
              <dt>Reasoning API credentials</dt><dd>{health.reasoning_configured ? 'Configured' : 'Missing'}</dd>
              <dt>Law corpus</dt><dd>{health.rag.corpus_available ? 'Available' : 'Missing'} · {health.rag.mode}</dd>
              <dt>Upload limit</dt><dd>{health.max_upload_mb} MB</dd>
            </dl>
            {health.video_vlm.error && <p className="text-amber-300">{health.video_vlm.error}</p>}
            <p className="text-slate-400">Configuration status does not verify model loading or provider connectivity. Configure credentials and corpus paths in the server .env file, then restart the API.</p>
          </>}
        </CardContent>
      </Card>
      <p className="text-sm text-slate-400">Uploaded videos, intermediate results, and reports are stored on the backend. API video mode sends sampled frames to the configured vision provider; reasoning sends descriptions and retrieved passages to the reasoning provider. Manage saved analyses in <Link to="/history" className="text-primary underline">History</Link>.</p>
    </div>
  );
}
