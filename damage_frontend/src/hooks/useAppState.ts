import { useCallback, useEffect, useState } from 'react'
import { getConfig } from '../services/api'
import type { AnalysisRow, AnalyzeResponse, ConfigResponse, CostResponse } from '../types'

type Step = 'upload' | 'photos' | 'review' | 'cost'

interface AppState {
  step: Step
  analysisResult: AnalyzeResponse | null
  editedRows: AnalysisRow[]
  costResult: CostResponse | null
  costStale: boolean
  loading: boolean
  error: string | null
  config: ConfigResponse | null
}

export function useAppState() {
  const [state, setState] = useState<AppState>({
    step: 'upload',
    analysisResult: null,
    editedRows: [],
    costResult: null,
    costStale: false,
    loading: false,
    error: null,
    config: null,
  })

  const loadConfig = useCallback(async () => {
    try {
      const cfg = await getConfig()
      setState((s) => ({ ...s, config: cfg }))
    } catch {
      // non-fatal — UI will degrade gracefully
    }
  }, [])

  useEffect(() => {
    loadConfig()
  }, [loadConfig])

  const goToStep = useCallback((step: Step) => {
    setState((s) => ({ ...s, step }))
  }, [])

  const setAnalysisResult = useCallback((result: AnalyzeResponse) => {
    setState((s) => ({
      ...s,
      analysisResult: result,
      editedRows: result.rows,
      costResult: null,
      costStale: false,
    }))
  }, [])

  const setEditedRows = useCallback((rows: AnalysisRow[]) => {
    setState((s) => ({
      ...s,
      editedRows: rows,
      costStale: s.costResult !== null ? true : s.costStale,
    }))
  }, [])

  const setCostResult = useCallback((result: CostResponse) => {
    setState((s) => ({ ...s, costResult: result, costStale: false }))
  }, [])

  const setLoading = useCallback((loading: boolean) => {
    setState((s) => ({ ...s, loading }))
  }, [])

  const setError = useCallback((error: string | null) => {
    setState((s) => ({ ...s, error }))
  }, [])

  return {
    ...state,
    goToStep,
    setAnalysisResult,
    setEditedRows,
    setCostResult,
    setLoading,
    setError,
    loadConfig,
  }
}
