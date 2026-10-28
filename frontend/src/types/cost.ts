export type ModelSource = 'api' | 'local' | 'compute';

export interface ModelCostRow {
  model_id: string;
  source: ModelSource;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

export interface CostGrandTotal {
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

export interface CostAnalysis {
  note: string;
  models: ModelCostRow[];
  grand_total: CostGrandTotal;
}
