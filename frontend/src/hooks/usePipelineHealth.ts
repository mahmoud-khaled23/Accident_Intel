import { useEffect, useState } from 'react';
import { get } from '@/services/apiClient';
import type { PipelineHealth } from '@/types/pipeline';

export function usePipelineHealth() {
  const [health, setHealth] = useState<PipelineHealth | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const value = await get<PipelineHealth>('/api/health');
        if (!cancelled) { setHealth(value); setError(null); }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'API unavailable.');
      }
    }
    void refresh();
    const timer = setInterval(() => void refresh(), 30_000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);
  return { health, error };
}
