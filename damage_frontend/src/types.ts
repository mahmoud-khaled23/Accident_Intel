export interface ImageUrls {
  original: string
  combined: string
  parts: string
  damage: string
}

export interface DamageOnPart {
  damage_type: string
  part: string
  confidence: number
  coverage: number
  area_ratio: number
  severity: string
  severity_source: string
  part_source: string
  bbox: number[]
}

export interface PhotoResult {
  photo_id: string
  usable: boolean
  notes: string[]
  parts_found: string[]
  items: DamageOnPart[]
  image_urls: ImageUrls
}

export interface FusedDamage {
  part: string
  damage_type: string
  severity: string
  count: number
  max_confidence: number
  photos: string[]
  needs_review: boolean
  review_reasons: string[]
}

export interface AnalysisRow {
  part: string
  damage_type: string
  severity: string
  count: number
  confidence: number | null
  needs_review: boolean
  reviewed: boolean
}

export interface AnalyzeResponse {
  analysis_id: string
  case_id: string
  warnings: string[]
  photos: PhotoResult[]
  fused: FusedDamage[]
  rows: AnalysisRow[]
}

export interface CostLine {
  part: string
  action: string
  damages: string[]
  parts_min: number
  parts_max: number
  labor_min: number
  labor_max: number
  paint_min: number
  paint_max: number
  total_min: number
  total_max: number
  price_source: string
  needs_review: boolean
  notes: string[]
}

export interface CarInfo {
  make: string
  model: string
  year: number
  tier: string
}

export interface CostReport {
  car: CarInfo
  currency: string
  lines: CostLine[]
  total_min: number
  total_max: number
  unconfirmed_min: number
  unconfirmed_max: number
  unpriced: string[]
  notes: string[]
}

export interface CostResponse {
  cost: CostReport
  warnings: string[]
}

export interface ConfigResponse {
  part_choices: string[]
  damage_types: string[]
  severities: string[]
  tiers: string[]
}
