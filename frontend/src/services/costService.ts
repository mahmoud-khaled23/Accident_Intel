import { get } from './apiClient';
import type { CostAnalysis } from '../types/cost';

export function getCostAnalysis(): Promise<CostAnalysis> {
  return get<CostAnalysis>('/api/cost/analysis');
}
