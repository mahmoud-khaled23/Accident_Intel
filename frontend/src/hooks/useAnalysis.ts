// Custom hooks for accident analysis data — wraps accidentService calls

import { useState, useEffect, useCallback, useRef } from 'react';
import type { AccidentRecord, VehicleTrack, VLMAnalysis, LiabilityAssessment } from '../types/accident';
import type { TrafficLaw } from '../types/law';
import type { PaginatedResponse } from '../types/common';
import {
  getAccidents,
  getAccident,
  uploadVideo,
  triggerAnalysis,
  pollStatus,
  getAccidentFull,
  getRetrievedLaws,
} from '../services/accidentService';
import { isMockMode, mockDelay } from '../services/apiClient';
import { MOCK_ACCIDENTS } from '../mocks/accidentMocks';
import { MOCK_TRAFFIC_LAWS } from '../mocks/lawMocks';

// ---------------------------------------------------------------------------
// useAccidents — paginated list of all accident records
// ---------------------------------------------------------------------------

export interface UseAccidentsResult {
  accidents: AccidentRecord[];
  total: number;
  totalPages: number;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useAccidents(
  page: number = 1,
  pageSize: number = 20,
): UseAccidentsResult {
  const [data, setData] = useState<PaginatedResponse<AccidentRecord> | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState<number>(0);

  const refetch = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getAccidents(page, pageSize)
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load analyses');
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [page, pageSize, tick]);

  return {
    accidents: data?.items ?? [],
    total: data?.total ?? 0,
    totalPages: data?.totalPages ?? 1,
    loading,
    error,
    refetch,
  };
}

// ---------------------------------------------------------------------------
// useAnalysis — single accident record by ID
// ---------------------------------------------------------------------------

export interface UseAnalysisResult {
  accident: AccidentRecord | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useAnalysis(id?: string): UseAnalysisResult {
  const [accident, setAccident] = useState<AccidentRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(Boolean(id));
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState<number>(0);

  const refetch = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!id) {
      setAccident(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    getAccident(id)
      .then((result) => {
        if (!cancelled) {
          setAccident(result);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : `Failed to load analysis ${id}`);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id, tick]);

  return { accident, loading, error, refetch };
}

// ---------------------------------------------------------------------------
// Pipeline step type
// ---------------------------------------------------------------------------

export type NewAnalysisStep =
  | 'idle'
  | 'uploading'
  | 'analyzing'
  | 'tracking'
  | 'vlm_analysis'
  | 'law_retrieval'
  | 'reasoning'
  | 'completed'
  | 'failed';

const STEP_MESSAGES: Record<string, string> = {
  uploading: 'Uploading video…',
  analyzing: 'Starting analysis pipeline…',
  processing: 'Starting pipeline…',
  tracking: 'Running vehicle tracking…',
  vlm_analysis: 'Analyzing video with VLM…',
  law_retrieval: 'Retrieving traffic laws…',
  reasoning: 'Generating liability report…',
  completed: 'Analysis complete.',
  failed: 'Analysis failed.',
};

export interface UseNewAnalysisResult {
  step: NewAnalysisStep;
  statusMessage: string;
  accidentId: string | null;
  trackingData: VehicleTrack[] | null;
  vlmAnalysis: VLMAnalysis | null;
  laws: TrafficLaw[] | null;
  liabilityReport: LiabilityAssessment | null;
  error: string | null;
  startAnalysis: (
    file: File,
    metadata: {
      location?: string;
      incidentDate?: string;
      description?: string;
      notes?: string;
    },
  ) => Promise<void>;
  reset: () => void;
}

// ---------------------------------------------------------------------------
// useNewAnalysis — manages the full upload→analyze→poll→complete flow
// ---------------------------------------------------------------------------

export function useNewAnalysis(): UseNewAnalysisResult {
  const [step, setStep] = useState<NewAnalysisStep>('idle');
  const [accidentId, setAccidentId] = useState<string | null>(null);
  const [trackingData, setTrackingData] = useState<VehicleTrack[] | null>(null);
  const [vlmAnalysis, setVlmAnalysis] = useState<VLMAnalysis | null>(null);
  const [laws, setLaws] = useState<TrafficLaw[] | null>(null);
  const [liabilityReport, setLiabilityReport] = useState<LiabilityAssessment | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Keep polling interval ref so we can clear it on unmount or completion
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const statusMessage = STEP_MESSAGES[step] ?? '';

  function clearPoll() {
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => clearPoll();
  }, []);

  function reset() {
    clearPoll();
    setStep('idle');
    setAccidentId(null);
    setTrackingData(null);
    setVlmAnalysis(null);
    setLaws(null);
    setLiabilityReport(null);
    setError(null);
  }

  // -------------------------------------------------------------------
  // Real-mode: fetch all results once pipeline completes
  // -------------------------------------------------------------------
  async function fetchCompletedResults(id: string) {
    try {
      const [fullRecord, retrievedLaws] = await Promise.all([
        getAccidentFull(id),
        getRetrievedLaws(id),
      ]);
      setTrackingData(fullRecord.tracks ?? []);
      setVlmAnalysis(fullRecord.vlmAnalysis ?? null);
      setLaws(retrievedLaws);
      setLiabilityReport(fullRecord.liabilityAssessment ?? null);
      setStep('completed');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load results';
      setError(msg);
      setStep('failed');
    }
  }

  // -------------------------------------------------------------------
  // Real-mode: start polling
  // -------------------------------------------------------------------
  function startPolling(id: string) {
    clearPoll();
    pollIntervalRef.current = setInterval(() => {
      pollStatus(id)
        .then((result) => {
          const backendStatus = result.status;
          const uiStep = (backendStatus as NewAnalysisStep) in STEP_MESSAGES
            ? (backendStatus as NewAnalysisStep)
            : 'analyzing';
          setStep(uiStep);

          if (backendStatus === 'completed') {
            clearPoll();
            void fetchCompletedResults(id);
          } else if (backendStatus === 'failed') {
            clearPoll();
            setError(result.error ?? 'Analysis pipeline failed');
            setStep('failed');
          }
        })
        .catch((err: unknown) => {
          // Don't stop polling on transient network errors — just skip this tick
          console.warn('pollStatus error:', err);
        });
    }, 3000);
  }

  // -------------------------------------------------------------------
  // startAnalysis — returns a Promise that resolves once the upload
  // (and trigger) succeed so the caller can safely call advance().
  // Rejects on upload/trigger failure so the caller can stay on Step 1.
  // -------------------------------------------------------------------
  function startAnalysis(
    file: File,
    metadata: {
      location?: string;
      incidentDate?: string;
      description?: string;
      notes?: string;
    },
  ): Promise<void> {
    if (isMockMode) {
      // Mock flow: simulate upload step, then continue asynchronously
      setStep('uploading');
      return (async () => {
        await mockDelay(800);
        // Resolve here — the caller calls advance(); the rest runs in bg
        void (async () => {
          try {
            setStep('tracking');
            await mockDelay(1000);
            setStep('vlm_analysis');
            await mockDelay(800);
            setStep('law_retrieval');
            await mockDelay(600);
            setStep('reasoning');
            await mockDelay(800);

            const mockRecord = MOCK_ACCIDENTS[0];
            const mockId = `MOCK-${String(Date.now()).slice(-4)}`;
            setAccidentId(mockId);
            setTrackingData(mockRecord.tracks ?? null);
            setVlmAnalysis(mockRecord.vlmAnalysis ?? null);
            setLaws(MOCK_TRAFFIC_LAWS.slice(0, 4));
            setLiabilityReport(mockRecord.liabilityAssessment ?? null);
            setStep('completed');
          } catch (err) {
            setError(err instanceof Error ? err.message : 'Mock analysis failed');
            setStep('failed');
          }
        })();
      })();
    }

    // Real mode — resolve only after upload + triggerAnalysis succeed
    setStep('uploading');
    return (async () => {
      const record = await uploadVideo(file, metadata);
      setAccidentId(record.id);
      setStep('analyzing');

      await triggerAnalysis(record.id);
      startPolling(record.id);
      // Promise resolves here — caller can safely call advance()
    })().catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : 'Upload or trigger failed';
      setError(msg);
      setStep('failed');
      throw err; // re-throw so caller stays on Step 1
    });
  }

  return {
    step,
    statusMessage,
    accidentId,
    trackingData,
    vlmAnalysis,
    laws,
    liabilityReport,
    error,
    startAnalysis,
    reset,
  };
}
