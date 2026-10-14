// Accident analysis service — mock mode or real FastAPI backend
// DEMO DATA — replace MOCK_* imports with real API calls when backend is connected

import type {
  AccidentRecord,
  VehicleTrack,
  TrackingFrame,
  VLMAnalysis,
  LiabilityAssessment,
  VehicleAnalysis,
  Violation,
  Evidence,
} from '../types/accident';
import type { TrafficLaw } from '../types/law';
import type { PaginatedResponse, AnalysisStatus } from '../types/common';
import {
  MOCK_ACCIDENTS,
  tracks001,
  vlmAnalysis001,
  liabilityAssessment001,
} from '../mocks/accidentMocks';
import { MOCK_TRAFFIC_LAWS } from '../mocks/lawMocks';
import { isMockMode, mockDelay, get, post, postForm, del } from './apiClient';

// ---------------------------------------------------------------------------
// Backend response shape (snake_case) — local interface only, not exported
// ---------------------------------------------------------------------------

interface BackendTrackingFrame {
  timestamp?: number;
  frame?: number;
  vehicle_id?: string;
  bbox?: { x: number; y: number; width: number; height: number } | number[];
  confidence?: number;
}

interface BackendVehicleTrajectory {
  id?: number | string;
  vehicle_id?: number | string;
  class?: string;
  frames?: BackendTrackingFrame[];
  speed?: number;
}

interface BackendTrackingData {
  vehicle_count?: number;
  fps?: number;
  width?: number;
  height?: number;
  trajectories?: BackendVehicleTrajectory[];
}

interface BackendLaw {
  section_id?: string;
  title?: string;
  content?: string;
  score?: number;
  metadata?: { jurisdiction?: string };
}

interface BackendVehicleAssessment {
  id?: number | string;
  role?: string;
  violations?: string[];
}

interface BackendReport {
  overall_assessment?: string;
  confidence?: number;
  overall_level?: LiabilityAssessment['overallLevel'];
  limitations?: string[];
  vehicles?: BackendVehicleAssessment[];
  violations?: Array<{
    description?: string;
    law_section?: string;
    severity?: string;
  }>;
}

interface BackendAccidentRecord {
  id: string;
  status: string;
  location?: string;
  date_time?: string;
  description?: string;
  notes?: string;
  created_at?: string;
  video_filename?: string;
  tracking_data?: BackendTrackingData | null;
  vlm_description?: string | null;
  laws?: BackendLaw[] | null;
  report?: BackendReport | null;
  error?: string | null;
  failed_stage?: string | null;
  is_demo_data?: boolean;
  video_metadata?: AccidentRecord['videoMetadata'];
}

// ---------------------------------------------------------------------------
// Color palette for vehicle tracks
// ---------------------------------------------------------------------------

const TRACK_COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#f59e0b',
  '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16',
];

// ---------------------------------------------------------------------------
// Backend → frontend mappers
// ---------------------------------------------------------------------------

function mapTrackingData(raw: BackendTrackingData | null | undefined): {
  tracks: VehicleTrack[];
  vehicleCount: number;
} {
  if (!raw) return { tracks: [], vehicleCount: 0 };

  const fps = raw.fps && raw.fps > 0 ? raw.fps : 25;
  const trajectories = raw.trajectories ?? [];

  const tracks: VehicleTrack[] = trajectories.map((traj, idx) => {
    const vehicleId = String(traj.vehicle_id ?? traj.id ?? idx + 1);
    const frames: TrackingFrame[] = (traj.frames ?? []).map((f) => {
      const rawFrame = f as BackendTrackingFrame & { frame?: number };
      const ts =
        typeof rawFrame.timestamp === 'number'
          ? rawFrame.timestamp
          : typeof rawFrame.frame === 'number' && fps > 0
            ? rawFrame.frame / fps
            : 0;

      let bbox = { x: 0, y: 0, width: 0, height: 0 };
      if (Array.isArray(f.bbox) && f.bbox.length === 4) {
        const [x1, y1, x2, y2] = f.bbox;
        bbox = {
          x: x1 / (raw.width || 1),
          y: y1 / (raw.height || 1),
          width: Math.max(0, x2 - x1) / (raw.width || 1),
          height: Math.max(0, y2 - y1) / (raw.height || 1),
        };
      } else if (f.bbox && typeof f.bbox === 'object') {
        bbox = f.bbox as { x: number; y: number; width: number; height: number };
      }

      return {
        timestamp: parseFloat(ts.toFixed(2)),
        vehicleId,
        bbox,
        confidence: f.confidence ?? 0,
      };
    });

    const timestamps = frames.map((f) => f.timestamp);
    const confidences = frames.map((f) => f.confidence);
    const meanConfidence =
      confidences.length > 0
        ? confidences.reduce((a, b) => a + b, 0) / confidences.length
        : 0;

    const firstSeen = timestamps.reduce((a, b) => Math.min(a, b), timestamps[0] ?? 0);
    const lastSeen = timestamps.reduce((a, b) => Math.max(a, b), 0);

    return {
      id: vehicleId,
      label: `Vehicle ${vehicleId}`,
      class: traj.class ?? 'vehicle',
      color: TRACK_COLORS[idx % TRACK_COLORS.length],
      speed: traj.speed ?? undefined,
      firstSeen,
      lastSeen,
      confidence: meanConfidence,
      events: [],
      frames,
    };
  });

  return {
    tracks,
    vehicleCount: raw.vehicle_count ?? tracks.length,
  };
}

function mapVLMDescription(text: string | null | undefined): VLMAnalysis {
  const raw = text?.trim() ?? '';
  const section = (name: string): string => {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const match = raw.match(new RegExp(`(?:^|\\n)\\s*(?:#+\\s*)?${escaped}\\s*:?[\\s\\r\\n]*([\\s\\S]*?)(?=\\n\\s*(?:#+\\s*)?[A-Z][A-Za-z &/]+\\s*:|$)`, 'i'));
    return match?.[1]?.trim() ?? '';
  };

  const scene = section('Scene Overview');
  const events = section('Chronological Events') || section('Timeline');
  const collision = section('Collision Evidence') || section('Collision Description');
  const finalState = section('Final Observed State') || section('Final State');
  const uncertainties = section('Uncertainties and Limitations') || section('Unclear Details');

  // The VLM's source output is prose. This adapter exposes the original text
  // in the existing UI without inventing structured claims or timestamps.
  return {
    summary: scene || raw,
    timeline: events ? [{ type: 'fact', content: events }] : [],
    vehicleActions: {},
    collisionDescription: collision || finalState,
    trafficSignals: '',
    roadContext: scene,
    uncertainties: uncertainties
      ? uncertainties.split(/\n|(?:^|\\s)[•-]\\s+/).map((item) => item.trim()).filter(Boolean)
      : [],
  };
}

function mapReport(raw: BackendReport | null | undefined): LiabilityAssessment | undefined {
  if (!raw) return undefined;

  const vehicleAnalyses: VehicleAnalysis[] = (raw.vehicles ?? []).map((v) => {
    const vid = String(v.id ?? '');
    const role = v.role ?? '';
    const liabilityLevel =
      role === 'primary_at_fault'
        ? 'potential_responsibility'
        : role === 'shared_responsibility' || role === 'secondary'
        ? 'shared_responsibility'
        : 'insufficient_evidence';

    return {
      vehicleId: vid,
      vehicleLabel: `Vehicle ${vid}`,
      liabilityLevel: liabilityLevel as VehicleAnalysis['liabilityLevel'],
      summary: v.violations?.join('; ') ?? '',
      contributingActions: v.violations ?? [],
      mitigatingFactors: [],
      evidence: [],
    };
  });

  const violations: Violation[] = (raw.violations ?? []).map((v, idx) => ({
    id: `violation-${idx}`,
    vehicleId: '',
    title: v.description ?? '',
    description: v.description ?? '',
    lawSection: v.law_section,
    severity: (v.severity as Violation['severity']) ?? 'minor',
    evidence: [] as Evidence[],
  }));

  return {
    overallAssessment: raw.overall_assessment ?? '',
    overallLevel: raw.overall_level && ['potential_responsibility', 'shared_responsibility', 'insufficient_evidence'].includes(raw.overall_level) ? raw.overall_level : 'insufficient_evidence',
    vehicleAnalyses,
    violations,
    conflictingEvidence: [],
    missingInfo: [],
    confidenceLevel: raw.confidence ?? 0,
    limitations: raw.limitations ?? [],
  };

}

function mapBackendRecord(raw: BackendAccidentRecord): AccidentRecord {
  const { tracks, vehicleCount } = mapTrackingData(raw.tracking_data);
  const vlmAnalysis = raw.vlm_description != null
    ? mapVLMDescription(raw.vlm_description)
    : undefined;
  const liabilityAssessment = mapReport(raw.report);
  const liabilityStatus: AccidentRecord['liabilityStatus'] =
    liabilityAssessment ? (liabilityAssessment.overallLevel === 'insufficient_evidence' ? 'insufficient_evidence' : 'assessed') : 'pending';

  const validStatuses: AnalysisStatus[] = [
    'created', 'processing', 'uploaded', 'tracking', 'vlm_analysis', 'law_retrieval',
    'reasoning', 'completed', 'failed',
  ];
  const rawStatus = raw.status as AnalysisStatus;
  const safeStatus: AnalysisStatus = validStatuses.includes(rawStatus) ? rawStatus : 'failed';

  return {
    id: raw.id,
    videoFilename: raw.video_filename ?? raw.id,
    uploadDate: raw.created_at ?? new Date().toISOString(),
    location: raw.location || undefined,
    incidentDate: raw.date_time || undefined,
    description: raw.description || undefined,
    notes: raw.notes || undefined,
    status: safeStatus,
    vehicleCount,
    liabilityStatus,
    tracks: tracks.length > 0 ? tracks : undefined,
    vlmAnalysis,
    liabilityAssessment,
    isDemoData: raw.is_demo_data ?? false,
    error: raw.error || undefined,
    failedStage: raw.failed_stage || undefined,
    videoMetadata: raw.video_metadata,
  };
}

function mapBackendLaws(rawLaws: BackendLaw[] | null | undefined): TrafficLaw[] {
  if (!rawLaws) return [];
  return rawLaws.map((l, idx) => ({
    id: l.section_id ?? `law-${idx}`,
    title: l.title ?? '',
    jurisdiction: l.metadata?.jurisdiction ?? 'NYS VTL',
    section: l.section_id ?? '',
    excerpt: l.content ?? '',
    fullText: l.content ?? '',
    // Use source URL from metadata when available; undefined lets the UI hide the link
    sourceUrl: ((l.metadata as Record<string, unknown>)?.source_url as string | undefined) || undefined,
    topics: [],
    relevanceScore: l.score,
  }));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** List all accident analyses with optional pagination */
export async function getAccidents(
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedResponse<AccidentRecord>> {
  if (isMockMode) {
    await mockDelay(500);
    const start = (page - 1) * pageSize;
    const items = MOCK_ACCIDENTS.slice(start, start + pageSize);
    return {
      items,
      total: MOCK_ACCIDENTS.length,
      page,
      pageSize,
      totalPages: Math.ceil(MOCK_ACCIDENTS.length / pageSize),
    };
  }
  const raw = await get<{
    items: BackendAccidentRecord[];
    total: number;
    page: number;
    limit: number;
    page_size?: number;
  }>('/api/accidents/', { page, page_size: pageSize });

  const effectivePageSize = raw.page_size ?? raw.limit;
  return {
    items: raw.items.map(mapBackendRecord),
    total: raw.total,
    page: raw.page,
    pageSize: effectivePageSize,
    totalPages: Math.ceil(raw.total / effectivePageSize),
  };
}

/** Fetch a single accident record by ID */
export async function getAccident(id: string): Promise<AccidentRecord> {
  if (isMockMode) {
    await mockDelay(400);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    if (!record) throw new Error(`Accident ${id} not found`);
    return record;
  }
  const raw = await get<BackendAccidentRecord>(`/api/accidents/${id}`);
  return mapBackendRecord(raw);
}

/** Fetch full accident record (same as getAccident, provided for hook convenience) */
export async function getAccidentFull(id: string): Promise<AccidentRecord> {
  return getAccident(id);
}

/** Upload a video file and associated metadata to start a new analysis */
export async function uploadVideo(
  file: File,
  metadata: {
    location?: string;
    incidentDate?: string;
    description?: string;
    notes?: string;
  },
): Promise<AccidentRecord> {
  if (isMockMode) {
    await mockDelay(800);
    const newRecord: AccidentRecord = {
      id: `ACC-${String(Date.now()).slice(-4)}`,
      videoFilename: file.name,
      uploadDate: new Date().toISOString(),
      location: metadata.location,
      incidentDate: metadata.incidentDate,
      description: metadata.description,
      notes: metadata.notes,
      status: 'uploaded',
      vehicleCount: 0,
      liabilityStatus: 'pending',
      isDemoData: true,
    };
    return newRecord;
  }
  const formData = new FormData();
  formData.append('file', file);
  if (metadata.location) formData.append('location', metadata.location);
  if (metadata.incidentDate) formData.append('incident_date', metadata.incidentDate);
  if (metadata.description) formData.append('description', metadata.description);
  if (metadata.notes) formData.append('notes', metadata.notes);
  const raw = await postForm<BackendAccidentRecord>('/api/accidents/upload', formData);
  // Attach the original filename since the server stores it internally
  raw.video_filename = raw.video_filename ?? file.name;
  return mapBackendRecord(raw);
}

/** Trigger the analysis pipeline for an already-uploaded accident */
export async function triggerAnalysis(id: string): Promise<{ status: string; message: string }> {
  if (isMockMode) {
    await mockDelay(300);
    return { status: 'processing', message: 'Mock analysis started' };
  }
  return post<{ status: string; message: string }>(`/api/accidents/${id}/analyze`);
}

/** Poll current pipeline status for an accident */
export async function pollStatus(id: string): Promise<{ id: string; status: string; error: string | null }> {
  if (isMockMode) {
    await mockDelay(200);
    return { id, status: 'completed', error: null };
  }
  return get<{ id: string; status: string; error: string | null }>(`/api/accidents/${id}/status`);
}

/** Get vehicle tracking data for an accident */
export async function getTrackingData(id: string): Promise<VehicleTrack[]> {
  if (isMockMode) {
    await mockDelay(600);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    return record?.tracks ?? tracks001;
  }
  const raw = await get<BackendTrackingData>(`/api/accidents/${id}/tracking`);
  return mapTrackingData(raw).tracks;
}

/** Get VLM (Vision-Language Model) analysis for an accident */
export async function getVLMAnalysis(id: string): Promise<VLMAnalysis> {
  if (isMockMode) {
    await mockDelay(700);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    return record?.vlmAnalysis ?? vlmAnalysis001;
  }
  const raw = await get<{ description: string }>(`/api/accidents/${id}/vlm`);
  return mapVLMDescription(raw.description);
}

/** Get NYC traffic laws retrieved via RAG for an accident */
export async function getRetrievedLaws(id: string): Promise<TrafficLaw[]> {
  if (isMockMode) {
    await mockDelay(500);
    // Return the 4 most relevant laws for the demo
    return MOCK_TRAFFIC_LAWS.slice(0, 4);
  }
  const raw = await get<{ laws: BackendLaw[] }>(`/api/accidents/${id}/laws`);
  return mapBackendLaws(raw.laws);
}

/** Get the liability assessment produced by the reasoning LLM */
export async function getLiabilityAssessment(id: string): Promise<LiabilityAssessment> {
  if (isMockMode) {
    await mockDelay(800);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    return record?.liabilityAssessment ?? liabilityAssessment001;
  }
  const raw = await get<{ report: BackendReport | null }>(`/api/accidents/${id}/liability`);
  const assessment = mapReport(raw.report);
  if (!assessment) throw new Error('Liability assessment not yet available');
  return assessment;
}

/** Delete an accident record */
export async function deleteAccident(id: string): Promise<void> {
  if (isMockMode) {
    await mockDelay(300);
    // In mock mode there is no persistent store — caller manages local state
    return;
  }
  await del<void>(`/api/accidents/${id}`);
}
