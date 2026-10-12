// Damage-domain type definitions for Accident-Intel

/** A single damaged component on the vehicle */
export interface DamagePart {
  id: string;
  name: string; // e.g. "Front Bumper"
  location: string; // e.g. "Front", "Rear", "Driver side"
  damageType: string; // e.g. "Dent", "Scratch", "Crack", "Shatter"
  severity: 'minor' | 'moderate' | 'severe';
  confidence: number; // AI detection confidence 0–1
  /** Normalized position on the vehicle image for overlay marker (0–1) */
  normalizedX?: number;
  normalizedY?: number;
  source: 'ai' | 'user';
}

/** A single line item in a repair cost breakdown */
export interface RepairLineItem {
  id: string;
  partId?: string; // links back to a DamagePart
  description: string;
  category: 'parts' | 'labor' | 'paint' | 'other' | 'tax';
  unitCost: number; // USD
  quantity: number;
  total: number; // unitCost * quantity
}

/** Aggregated repair cost estimate */
export interface RepairEstimate {
  subtotal: number; // USD, before tax
  tax: number; // USD
  total: number; // USD, after tax
  lowEstimate: number; // USD
  highEstimate: number; // USD
  currency: string; // "USD"
  egpEquivalent?: number; // EGP equivalent at time of estimate
  egpRate?: number; // exchange rate used
  assumptions: string[];
  limitations: string[];
}

/** Basic vehicle identification information */
export interface VehicleInfo {
  make: string; // e.g. "Toyota"
  model: string; // e.g. "Camry"
  year: number;
  color: string;
  mileage?: number; // km
  vin?: string;
  repairLocation?: string; // e.g. "Cairo", "Alexandria"
}

/** Full damage assessment record */
export interface DamageAssessmentRecord {
  id: string;
  sessionId: string;
  vehicleInfo: VehicleInfo;
  imageUrls: string[]; // base64 data URLs or placeholder paths
  damagedParts: DamagePart[];
  repairItems: RepairLineItem[];
  estimate: RepairEstimate;
  assessmentDate: string; // ISO 8601
  isDemoData?: boolean;
}
