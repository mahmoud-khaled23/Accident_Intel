export type VideoBackend = 'auto' | 'local' | 'openai';

export interface VideoModelInfo {
  requested: VideoBackend;
  backend: 'local' | 'openai' | 'mock';
  model: string | null;
  module: string | null;
  gpu_available: boolean;
  api_configured: boolean;
  ready: boolean;
  error: string | null;
}

export interface PipelineHealth {
  status: string;
  pipeline: Record<'tracking' | 'vlm' | 'rag' | 'reasoning', 'real' | 'demo'>;
  video_vlm: VideoModelInfo;
  max_upload_mb: number;
  rag: { mode: string; rerank: boolean; corpus_available: boolean };
  reasoning_configured: boolean;
}
