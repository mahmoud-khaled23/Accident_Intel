import axios from 'axios';

export interface EvaluationSummary {
  run_id: string;
  created_at_utc: string;
  stages: Record<
    string,
    {
      stage_name: string;
      status: 'SUCCESS' | 'PARTIAL' | 'FAILED' | 'SKIPPED' | 'BLOCKED';
      sample_count: number;
      duration_seconds: number;
      reason?: string | null;
      metrics: Record<string, any>;
      failure_case_count: number;
      failure_cases: any[];
      metadata: Record<string, any>;
    }
  >;
}

export interface EvaluationManifest {
  run_id: string;
  created_at_utc: string;
  git: {
    commit: string;
    branch: string;
    is_dirty: boolean;
  };
  environment: {
    python_version: string;
    platform: string;
    processor: string;
    torch_version: string | null;
    cuda_available: boolean;
    cuda_device: string | null;
  };
  datasets: Array<{
    name: string;
    file_path: string;
    sample_count: number;
    sha256: string;
  }>;
}

export interface LatestEvaluationResponse {
  status: 'available' | 'no_results';
  summary?: EvaluationSummary;
  manifest?: EvaluationManifest;
  message?: string;
}

export interface EvaluationRunHistoryItem {
  run_id: string;
  created_at_utc: string;
  stages: Record<string, string>;
  git_commit: string;
}

export async function getLatestEvaluation(): Promise<LatestEvaluationResponse> {
  const resp = await axios.get<LatestEvaluationResponse>('/api/evaluation/latest');
  return resp.data;
}

export async function getEvaluationHistory(): Promise<EvaluationRunHistoryItem[]> {
  const resp = await axios.get<EvaluationRunHistoryItem[]>('/api/evaluation/history');
  return resp.data;
}

export function getArtifactDownloadUrl(filename: string): string {
  return `/api/evaluation/artifacts/${encodeURIComponent(filename)}`;
}

