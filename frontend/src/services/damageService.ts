// Damage assessment service — mock mode or real FastAPI backend
// DEMO DATA — replace MOCK_* imports with real API calls when backend is connected

import type { DamageAssessmentRecord, VehicleInfo, DamagePart } from '../types/damage';
import { MOCK_DAMAGE_ASSESSMENTS } from '../mocks/damageMocks';
import { isMockMode, mockDelay, get, postForm, post } from './apiClient';

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
  const formData = new FormData();
  files.forEach((file) => formData.append('images', file));
  formData.append('make', vehicleInfo.make);
  formData.append('model', vehicleInfo.model);
  formData.append('year', String(vehicleInfo.year));
  formData.append('color', vehicleInfo.color);
  if (vehicleInfo.mileage !== undefined) formData.append('mileage', String(vehicleInfo.mileage));
  if (vehicleInfo.vin) formData.append('vin', vehicleInfo.vin);
  if (vehicleInfo.repairLocation) formData.append('repair_location', vehicleInfo.repairLocation);

  // This endpoint runs the real damage pipeline and already returns the UI's
  // DamageAssessmentRecord shape.
  return postForm<DamageAssessmentRecord>('/api/damage/upload', formData);
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
