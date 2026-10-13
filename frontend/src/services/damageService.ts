// Damage assessment service — mock mode or real FastAPI backend
// DEMO DATA — replace MOCK_* imports with real API calls when backend is connected

import type { DamageAssessmentRecord, VehicleInfo, DamagePart } from '../types/damage';
import { MOCK_DAMAGE_ASSESSMENTS } from '../mocks/damageMocks';
import { isMockMode, mockDelay, get, postForm, post } from './apiClient';

interface BackendCostLine {
  part?: string;
  action?: string;
  damages?: string[];
  total_min?: number;
  total_max?: number;
}

interface BackendDamageResponse {
  id: string;
  status: string;
  vehicle_info?: Record<string, unknown>;
  images?: string[];
  cost_report?: {
    total_min?: number;
    total_max?: number;
    currency?: string;
    lines?: BackendCostLine[];
    notes?: string[];
  } | null;
  created_at?: string;
}

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function titleCase(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function mapBackendAssessment(
  response: BackendDamageResponse,
  suppliedVehicleInfo: VehicleInfo,
): DamageAssessmentRecord {
  const report = response.cost_report ?? {};
  const lines = report.lines ?? [];
  const lowEstimate = report.total_min ?? 0;
  const highEstimate = report.total_max ?? lowEstimate;
  const total = (lowEstimate + highEstimate) / 2;

  return {
    id: response.id,
    sessionId: response.id,
    vehicleInfo: suppliedVehicleInfo,
    imageUrls: response.images ?? [],
    damagedParts: lines.map((line, index) => ({
      id: `${response.id}-part-${index}`,
      name: titleCase(line.part ?? 'Unknown part'),
      location: 'Not specified',
      damageType: (line.damages ?? []).join(', ') || 'Damage detected',
      severity: /severe/i.test((line.damages ?? []).join(' '))
        ? 'severe'
        : /moderate/i.test((line.damages ?? []).join(' '))
          ? 'moderate'
          : 'minor',
      confidence: 0,
      source: 'ai',
    })),
    repairItems: lines.map((line, index) => {
      const itemLow = line.total_min ?? 0;
      const itemHigh = line.total_max ?? itemLow;
      const itemTotal = (itemLow + itemHigh) / 2;
      return {
        id: `${response.id}-repair-${index}`,
        description: `${titleCase(line.part ?? 'Unknown part')}: ${titleCase(line.action ?? 'repair')}`,
        category: 'other' as const,
        unitCost: itemTotal,
        quantity: 1,
        total: itemTotal,
      };
    }),
    estimate: {
      subtotal: total,
      tax: 0,
      total,
      lowEstimate,
      highEstimate,
      currency: report.currency ?? 'EGP',
      assumptions: report.notes ?? [],
      limitations: [],
    },
    assessmentDate: response.created_at ?? new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** List all damage assessments */
export async function getDamageAssessments(): Promise<DamageAssessmentRecord[]> {
  if (isMockMode) {
    await mockDelay(400);
    return MOCK_DAMAGE_ASSESSMENTS;
  }
  return get<DamageAssessmentRecord[]>('/api/damage');
}

/** Fetch a single damage assessment by ID */
export async function getDamageAssessment(id: string): Promise<DamageAssessmentRecord> {
  if (isMockMode) {
    await mockDelay(400);
    const record = MOCK_DAMAGE_ASSESSMENTS.find((d) => d.id === id);
    if (!record) throw new Error(`Damage assessment ${id} not found`);
    return record;
  }
  return get<DamageAssessmentRecord>(`/api/damage/${id}`);
}

/** Upload damage images and vehicle info to start a new assessment */
export async function uploadDamageImages(
  files: File[],
  vehicleInfo: VehicleInfo,
): Promise<DamageAssessmentRecord> {
  if (isMockMode) {
    await mockDelay(900);
    // Return the first mock assessment with updated vehicle info and new IDs
    const base = MOCK_DAMAGE_ASSESSMENTS[0];
    const mockResult: DamageAssessmentRecord = {
      ...base,
      id: `DMG-${String(Date.now()).slice(-4)}`,
      sessionId: `session-${String(Date.now()).slice(-6)}`,
      vehicleInfo,
      imageUrls: files.map((f) => URL.createObjectURL(f)),
      assessmentDate: new Date().toISOString(),
      isDemoData: true,
    };
    return mockResult;
  }
  // The backend deliberately separates creation, image upload, and estimation.
  // Keep that workflow in one client method so the page can remain simple.
  const created = await post<BackendDamageResponse>('/api/damage-assessments/', {
    make: vehicleInfo.make,
    model: vehicleInfo.model,
    year: vehicleInfo.year,
    tier: 'mid',
    currency: 'EGP',
  });

  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  await postForm<{ message: string }>(`/api/damage-assessments/${created.id}/images`, formData);
  await post<{ status: string }>(`/api/damage-assessments/${created.id}/estimate`);

  // Estimation runs in a FastAPI background task. Poll until its cost report is ready.
  for (let attempt = 0; attempt < 120; attempt += 1) {
    await wait(1_000);
    const current = await get<BackendDamageResponse>(`/api/damage-assessments/${created.id}`);
    if (current.status === 'completed' && current.cost_report) {
      return mapBackendAssessment(current, vehicleInfo);
    }
    if (current.status === 'failed') {
      throw new Error('Damage assessment failed on the server. Check the API logs for details.');
    }
  }
  throw new Error('Damage assessment is taking longer than expected. Please try again shortly.');
}

/** Apply user-supplied corrections to an existing damage assessment */
export async function updateDamageCorrections(
  id: string,
  corrections: Partial<DamagePart>[],
): Promise<DamageAssessmentRecord> {
  if (isMockMode) {
    await mockDelay(400);
    const record = MOCK_DAMAGE_ASSESSMENTS.find((d) => d.id === id);
    if (!record) throw new Error(`Damage assessment ${id} not found`);
    // Apply corrections to existing parts
    const updatedParts: DamagePart[] = record.damagedParts.map((part) => {
      const correction = corrections.find((c) => c.id === part.id);
      return correction ? { ...part, ...correction, source: 'user' as const } : part;
    });
    return { ...record, damagedParts: updatedParts };
  }
  return post<DamageAssessmentRecord>(`/api/damage/${id}/corrections`, corrections);
}
