// Accident-domain type definitions for Accident-Intel

import type { AnalysisStatus, LiabilityLevel } from './common';

/** Normalized bounding box (0–1 relative to frame dimensions) */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A single vehicle's bounding box at a specific video timestamp */
export interface TrackingFrame {
  timestamp: number; // seconds from video start
  vehicleId: string;
  bbox: BoundingBox;
  confidence: number; // 0–1
}

/** Aggregated tracking data for one vehicle across the entire video */
export interface VehicleTrack {
  id: string;
  label: string; // e.g. "Vehicle 1"
  class: string; // e.g. "sedan", "SUV", "truck"
  color: string; // CSS hex color for bounding-box overlay
  speed?: number; // estimated km/h at time of collision
  firstSeen: number; // timestamp in seconds
  lastSeen: number; // timestamp in seconds
  confidence: number; // mean tracking confidence 0–1
  events: string[]; // human-readable event descriptions
  frames: TrackingFrame[];
}

/** A single observation from the Vision-Language Model */
export interface VLMObservation {
  timestamp?: number; // seconds — optional link back to video
  type: 'fact' | 'interpretation';
  content: string;
}

/** Full VLM analysis output for an accident */
export interface VLMAnalysis {
  summary: string;
  timeline: VLMObservation[];
  vehicleActions: Record<string, string>; // vehicleId → description
  collisionDescription: string;
  trafficSignals: string;
  roadContext: string;
  pedestrians?: string;
  uncertainties: string[];
}

/** Evidence item linking a finding to a source */
export interface Evidence {
  id: string;
  type: 'video_timestamp' | 'law_citation' | 'observation';
  description: string;
  timestamp?: number; // seconds, for video_timestamp type
  lawId?: string; // for law_citation type
}

/** A potential traffic law violation identified during reasoning */
export interface Violation {
  id: string;
  vehicleId: string;
  title: string;
  description: string;
  lawSection?: string;
  severity: 'minor' | 'moderate' | 'major';
  evidence: Evidence[];
}

/** Per-vehicle assessment from the reasoning LLM */
export interface VehicleAnalysis {
  vehicleId: string;
  vehicleLabel: string;
  liabilityLevel: LiabilityLevel;
  summary: string;
  contributingActions: string[];
  mitigatingFactors: string[];
  evidence: Evidence[];
}

/** Complete liability assessment from the reasoning step */
export interface LiabilityAssessment {
  overallAssessment: string;
  overallLevel: LiabilityLevel;
  vehicleAnalyses: VehicleAnalysis[];
  violations: Violation[];
  conflictingEvidence: string[];
  missingInfo: string[];
  confidenceLevel: number; // 0–1
  limitations: string[];
}

/** Top-level accident record stored in the system */
export interface AccidentRecord {
  id: string;
  videoFilename: string;
  uploadDate: string; // ISO 8601
  location?: string;
  incidentDate?: string; // ISO 8601
  description?: string;
  notes?: string;
  status: AnalysisStatus;
  vehicleCount: number;
  liabilityStatus: 'pending' | 'assessed' | 'insufficient_evidence';
  tracks?: VehicleTrack[];
  vlmAnalysis?: VLMAnalysis;
  liabilityAssessment?: LiabilityAssessment;
  isDemoData?: boolean;
  error?: string;
  failedStage?: string;
  videoMetadata?: { width: number; height: number; fps: number; duration_seconds: number };

}
