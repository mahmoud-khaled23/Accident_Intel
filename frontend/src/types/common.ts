// Shared enumerations and utility types for Accident-Intel

export type AnalysisStatus =
  | 'uploaded'
  | 'tracking'
  | 'vlm_analysis'
  | 'law_retrieval'
  | 'reasoning'
  | 'completed'
  | 'failed';

export type LiabilityLevel =
  | 'potential_responsibility'
  | 'shared_responsibility'
  | 'insufficient_evidence';

export type ConfidenceLevel = 'low' | 'medium' | 'high';

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  error?: string;
}

export interface LoadingState {
  isLoading: boolean;
  loadingMessage?: string;
}

export interface ErrorState {
  hasError: boolean;
  errorMessage?: string;
  errorCode?: string;
}

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}
