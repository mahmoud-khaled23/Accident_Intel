import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Loader2, Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { PageHeader } from '@/components/common/PageHeader';
import { VideoUploader } from '@/components/upload/VideoUploader';
import { VideoPreview } from '@/components/upload/VideoPreview';
import { uploadVideo, triggerAnalysis } from '@/services/accidentService';
import { isMockMode } from '@/services/apiClient';

export function NewAnalysisPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState({ location: '', incidentDate: '', description: '', notes: '' });
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'starting'>('idle');
  const [savedId, setSavedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = phase !== 'idle';

  async function start() {
    if (!file || busy) return;
    setError(null);
    // Demo mode is explicit and never pretends to analyze an uploaded file.
    if (isMockMode) {
      setError('Demo mode is enabled. Set VITE_USE_MOCK=false to analyze your video.');
      return;
    }
    try {
      let id = savedId;
      if (!id) {
        setPhase('uploading');
        const record = await uploadVideo(file, {
          ...metadata,
          incidentDate: metadata.incidentDate ? new Date(metadata.incidentDate).toISOString() : undefined,
        });
        id = record.id;
        setSavedId(id);
      }
      setPhase('starting');
      await triggerAnalysis(id);
      navigate(`/analysis/${id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start analysis.');
    } finally {
      setPhase('idle');
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <PageHeader title="New Video Analysis" description="Upload a video to track vehicles, analyze visible events, retrieve traffic laws, and generate a report." />
      <fieldset disabled={busy || Boolean(savedId)} className="space-y-6 disabled:opacity-70">
        <VideoUploader file={file} onFileSelect={(value) => { setFile(value); setError(null); }} onRemove={() => setFile(null)} />
        {file && <VideoPreview file={file} />}
        <Card>
          <CardHeader><CardTitle>Incident details (optional)</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-2"><Label htmlFor="location">Location</Label><Input id="location" value={metadata.location} onChange={(e) => setMetadata({ ...metadata, location: e.target.value })} /></div>
              <div className="space-y-2"><Label htmlFor="incidentDate">Incident date and time</Label><Input id="incidentDate" type="datetime-local" value={metadata.incidentDate} onChange={(e) => setMetadata({ ...metadata, incidentDate: e.target.value })} /></div>
            </div>
            <div className="space-y-2"><Label htmlFor="description">Description</Label><Textarea id="description" value={metadata.description} onChange={(e) => setMetadata({ ...metadata, description: e.target.value })} /></div>
            <div className="space-y-2"><Label htmlFor="notes">Investigation notes or questions</Label><Textarea id="notes" value={metadata.notes} onChange={(e) => setMetadata({ ...metadata, notes: e.target.value })} /></div>
          </CardContent>
        </Card>
      </fieldset>
      {error && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">{error}</div>}
      <div className="flex flex-wrap items-center gap-4">
        <Button size="lg" disabled={!file || busy} onClick={() => void start()}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {phase === 'uploading' ? 'Uploading video…' : phase === 'starting' ? 'Starting analysis…' : savedId ? 'Retry analysis' : 'Upload and analyze'}
        </Button>
        {savedId && <Link className="text-sm text-primary underline" to={`/analysis/${savedId}`}>Open saved analysis</Link>}
      </div>
      <p className="text-sm text-slate-400">Progress and results are saved in History. First use may take longer while models load.</p>
    </div>
  );
}
