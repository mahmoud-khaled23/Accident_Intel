// Accident analysis service — mock mode or real FastAPI backend
// DEMO DATA — replace MOCK_* imports with real API calls when backend is connected

import type {
  AccidentRecord,
  VehicleTrack,
  VLMAnalysis,
  LiabilityAssessment,
} from '../types/accident';
import type { TrafficLaw } from '../types/law';
import type { PaginatedResponse } from '../types/common';
import {
  MOCK_ACCIDENTS,
  tracks001,
  vlmAnalysis001,
  liabilityAssessment001,
} from '../mocks/accidentMocks';
import { MOCK_TRAFFIC_LAWS } from '../mocks/lawMocks';
import { isMockMode, mockDelay, get, post, postForm } from './apiClient';

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
  return get<PaginatedResponse<AccidentRecord>>('/api/accidents', {
    page,
    page_size: pageSize,
  });
}

/** Fetch a single accident record by ID */
export async function getAccident(id: string): Promise<AccidentRecord> {
  if (isMockMode) {
    await mockDelay(400);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    if (!record) throw new Error(`Accident ${id} not found`);
    return record;
  }
  return get<AccidentRecord>(`/api/accidents/${id}`);
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
  return postForm<AccidentRecord>('/api/accidents/upload', formData);
}

/** Get vehicle tracking data for an accident */
export async function getTrackingData(id: string): Promise<VehicleTrack[]> {
  if (isMockMode) {
    await mockDelay(600);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    return record?.tracks ?? tracks001;
  }
  return get<VehicleTrack[]>(`/api/accidents/${id}/tracking`);
}

/** Get VLM (Vision-Language Model) analysis for an accident */
export async function getVLMAnalysis(id: string): Promise<VLMAnalysis> {
  if (isMockMode) {
    await mockDelay(700);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    return record?.vlmAnalysis ?? vlmAnalysis001;
  }
  return get<VLMAnalysis>(`/api/accidents/${id}/vlm`);
}

/** Get NYC traffic laws retrieved via RAG for an accident */
export async function getRetrievedLaws(id: string): Promise<TrafficLaw[]> {
  if (isMockMode) {
    await mockDelay(500);
    // Return the 4 most relevant laws for the demo
    return MOCK_TRAFFIC_LAWS.slice(0, 4);
  }
  return get<TrafficLaw[]>(`/api/accidents/${id}/laws`);
}

/** Get the liability assessment produced by the reasoning LLM */
export async function getLiabilityAssessment(id: string): Promise<LiabilityAssessment> {
  if (isMockMode) {
    await mockDelay(800);
    const record = MOCK_ACCIDENTS.find((a) => a.id === id);
    return record?.liabilityAssessment ?? liabilityAssessment001;
  }
  return get<LiabilityAssessment>(`/api/accidents/${id}/liability`);
}

/** Delete an accident record */
export async function deleteAccident(id: string): Promise<void> {
  if (isMockMode) {
    await mockDelay(300);
    // In mock mode there is no persistent store — caller manages local state
    return;
  }
  await post<void>(`/api/accidents/${id}/delete`);
}
