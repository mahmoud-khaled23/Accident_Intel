// Custom hooks for accident analysis data — wraps accidentService calls

import { useState, useEffect, useCallback } from 'react';
import type { AccidentRecord } from '../types/accident';
import type { PaginatedResponse } from '../types/common';
import {
  getAccidents,
  getAccident,
} from '../services/accidentService';

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
