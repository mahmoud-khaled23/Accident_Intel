import axios, { AxiosError } from 'axios'
import type {
  AnalysisRow,
  AnalyzeResponse,
  ConfigResponse,
  CostResponse,
} from '../types'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  timeout: 120000,
})

function extractMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((d) => d.msg ?? String(d)).join('; ')
    return err.message
  }
  if (err instanceof Error) return err.message
  return String(err)
}

export async function analyzePhotos(formData: FormData): Promise<AnalyzeResponse> {
  try {
    const res = await client.post<AnalyzeResponse>('/api/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return res.data
  } catch (err) {
    throw new Error(extractMessage(err))
  }
}

export async function getConfig(): Promise<ConfigResponse> {
  try {
    const res = await client.get<ConfigResponse>('/api/config')
    return res.data
  } catch (err) {
    throw new Error(extractMessage(err))
  }
}

export async function estimateCost(body: {
  analysis_id: string
  rows: AnalysisRow[]
  make: string
  model_name: string
  year: number
  tier: string
  use_web: boolean
}): Promise<CostResponse> {
  try {
    const res = await client.post<CostResponse>('/api/cost-estimate', body)
    return res.data
  } catch (err) {
    throw new Error(extractMessage(err))
  }
}
