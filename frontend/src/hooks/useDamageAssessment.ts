// Custom hooks for damage assessment data — wraps damageService calls

import { useState, useEffect, useCallback } from 'react';
import type { DamageAssessmentRecord } from '../types/damage';
import {
  getDamageAssessments,
  getDamageAssessment,
} from '../services/damageService';

// ---------------------------------------------------------------------------
// useDamageAssessments — list of all damage assessment records
// ---------------------------------------------------------------------------

export interface UseDamageAssessmentsResult {
  assessments: DamageAssessmentRecord[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useDamageAssessments(): UseDamageAssessmentsResult {
  const [assessments, setAssessments] = useState<DamageAssessmentRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState<number>(0);

  const refetch = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getDamageAssessments()
      .then((result) => {
        if (!cancelled) {
          setAssessments(result);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : 'Failed to load damage assessments',
          );
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [tick]);

  return { assessments, loading, error, refetch };
}

// ---------------------------------------------------------------------------
// useDamageAssessment — single assessment by ID
// ---------------------------------------------------------------------------

export interface UseDamageAssessmentResult {
  assessment: DamageAssessmentRecord | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useDamageAssessment(id?: string): UseDamageAssessmentResult {
  const [assessment, setAssessment] = useState<DamageAssessmentRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(Boolean(id));
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState<number>(0);

  const refetch = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!id) {
      setAssessment(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    getDamageAssessment(id)
      .then((result) => {
        if (!cancelled) {
          setAssessment(result);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : `Failed to load damage assessment ${id}`,
          );
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id, tick]);

  return { assessment, loading, error, refetch };
}
