// DEMO DATA — replace with real API responses when backend is connected

import type {
  AccidentRecord,
  VehicleTrack,
  TrackingFrame,
  VLMAnalysis,
  LiabilityAssessment,
} from '../types/accident';

// ---------------------------------------------------------------------------
// Helper: generate a series of bounding-box frames for a vehicle
// ---------------------------------------------------------------------------
function makeFrames(
  vehicleId: string,
  startTs: number,
  endTs: number,
  startX: number,
  startY: number,
  dxPerSec: number,
  dyPerSec: number,
  count: number = 25,
): TrackingFrame[] {
  const frames: TrackingFrame[] = [];
  const duration = endTs - startTs;
  for (let i = 0; i < count; i++) {
    const t = startTs + (duration * i) / (count - 1);
    const elapsed = t - startTs;
    frames.push({
      timestamp: parseFloat(t.toFixed(2)),
      vehicleId,
      bbox: {
        x: Math.min(0.95, Math.max(0.01, startX + dxPerSec * elapsed)),
        y: Math.min(0.90, Math.max(0.01, startY + dyPerSec * elapsed)),
        width: 0.12,
        height: 0.08,
      },
      confidence: parseFloat((0.82 + Math.random() * 0.16).toFixed(3)),
    });
  }
  return frames;
}

// ---------------------------------------------------------------------------
// VLM Analysis — fully populated (used by ACC-001)
// ---------------------------------------------------------------------------
const vlmAnalysis001: VLMAnalysis = {
  summary:
    'The footage captures a two-vehicle collision at an urban intersection. Vehicle 1 (silver sedan) entered the intersection on a yellow-turning-red signal while Vehicle 2 (black SUV) was proceeding through on a green signal from the cross street. The impact occurred near the center of the intersection.',
  timeline: [
    {
      timestamp: 2.1,
      type: 'fact',
      content: 'Vehicle 1 (silver sedan) is visible approaching the intersection from the north on a clear road.',
    },
    {
      timestamp: 5.4,
      type: 'fact',
      content: 'Traffic signal for Vehicle 1\'s lane transitions from green to yellow.',
    },
    {
      timestamp: 7.8,
      type: 'fact',
      content: 'Vehicle 1 maintains speed and does not appear to brake before the stop line.',
    },
    {
      timestamp: 8.9,
      type: 'fact',
      content: 'Traffic signal for Vehicle 1\'s lane turns red. Vehicle 1 has not yet reached the stop line.',
    },
    {
      timestamp: 9.2,
      type: 'fact',
      content: 'Vehicle 2 (black SUV) enters the intersection from the west; its signal appears green.',
    },
    {
      timestamp: 9.7,
      type: 'interpretation',
      content: 'Vehicle 1 appears to accelerate slightly through the red signal, though lighting conditions make precise speed estimation uncertain.',
    },
    {
      timestamp: 10.3,
      type: 'fact',
      content: 'Vehicles 1 and 2 collide near the center of the intersection. Vehicle 1 strikes the front-right quarter of Vehicle 2.',
    },
    {
      timestamp: 10.3,
      type: 'interpretation',
      content: 'Based on post-impact trajectories, Vehicle 1 was traveling approximately 45–55 km/h and Vehicle 2 approximately 35–45 km/h at impact. These are estimates derived from vehicle movement and cannot be independently verified from this footage alone.',
    },
    {
      timestamp: 12.5,
      type: 'fact',
      content: 'Both vehicles come to rest. Vehicle 2 has rotated approximately 90 degrees clockwise from its original heading.',
    },
    {
      timestamp: 14.0,
      type: 'fact',
      content: 'No pedestrians are visible within the intersection at the time of impact.',
    },
  ],
  vehicleActions: {
    'v1': 'Approached intersection during yellow signal, failed to stop at red, continued into intersection and collided with Vehicle 2.',
    'v2': 'Entered intersection on green signal from perpendicular direction; no evasive action observed before impact.',
  },
  collisionDescription:
    'A T-bone style collision occurred when Vehicle 1 ran a red signal and struck the front-right quarter panel of Vehicle 2 near the center of the intersection. The impact angle was approximately 85 degrees. Vehicle 2 sustained significant front-right damage and rotated post-impact.',
  trafficSignals:
    'The signal controlling Vehicle 1\'s approach lane clearly shows red at the moment Vehicle 1 crosses the stop line. The signal controlling Vehicle 2\'s approach lane shows green throughout its approach and at impact. Signal timing is consistent with standard NYC intersection operation.',
  roadContext:
    'Urban four-way signalized intersection. Two lanes in each direction. Road surface is dry. Intersection is well-lit with overhead street lighting. No construction zones or unusual obstructions are visible.',
  pedestrians:
    'No pedestrians present in the intersection at the time of the collision. A pedestrian is visible on the northwest corner but does not enter the roadway.',
  uncertainties: [
    'Exact vehicle speeds cannot be confirmed without telemetry or calibrated reference markers.',
    'Whether Vehicle 1\'s driver perceived the red signal is unknown from visual footage alone.',
    'Visibility conditions for Vehicle 1\'s driver (sun glare, obscured signal) cannot be fully assessed from the camera angle.',
  ],
};

// ---------------------------------------------------------------------------
// Liability Assessment — fully populated (used by ACC-001)
// ---------------------------------------------------------------------------
const liabilityAssessment001: LiabilityAssessment = {
  overallAssessment:
    'Evidence from the footage and applicable NYC traffic regulations strongly suggests that Vehicle 1 bears primary responsibility for the collision by proceeding through a red signal. Vehicle 2 appears to have been lawfully operating with right of way.',
  overallLevel: 'potential_responsibility',
  vehicleAnalyses: [
    {
      vehicleId: 'v1',
      vehicleLabel: 'Vehicle 1 — Silver Sedan',
      liabilityLevel: 'potential_responsibility',
      summary:
        'Vehicle 1 crossed the stop line after the signal turned red and failed to yield to cross-traffic with right of way. This behavior is consistent with a violation of SAMPLE-VTL-1111 (traffic signals) and SAMPLE-VTL-1140 (right of way at intersection).',
      contributingActions: [
        'Failed to stop at red signal',
        'Entered intersection without right of way',
        'Did not appear to apply brakes before crossing stop line',
      ],
      mitigatingFactors: [
        'Possible signal visibility obstruction not determinable from footage',
        'Signal had just turned red — reaction time ambiguity exists',
      ],
      evidence: [
        {
          id: 'ev-001',
          type: 'video_timestamp',
          description: 'Signal turns red while Vehicle 1 is before the stop line',
          timestamp: 8.9,
        },
        {
          id: 'ev-002',
          type: 'video_timestamp',
          description: 'Vehicle 1 crosses stop line after red indication',
          timestamp: 9.1,
        },
        {
          id: 'ev-003',
          type: 'law_citation',
          description: 'VTL § SAMPLE-1111 — Obedience to traffic control devices',
          lawId: 'law-signals',
        },
      ],
    },
    {
      vehicleId: 'v2',
      vehicleLabel: 'Vehicle 2 — Black SUV',
      liabilityLevel: 'insufficient_evidence',
      summary:
        'Vehicle 2 entered the intersection on a green signal. No evidence of speeding, distraction, or failure to yield is observed. Vehicle 2 appears to have been operating lawfully.',
      contributingActions: [],
      mitigatingFactors: [
        'Signal shows green throughout Vehicle 2\'s approach',
        'No pre-impact evasive opportunity observable from footage',
      ],
      evidence: [
        {
          id: 'ev-004',
          type: 'video_timestamp',
          description: 'Vehicle 2 visible approaching on green signal',
          timestamp: 9.2,
        },
      ],
    },
  ],
  violations: [
    {
      id: 'viol-001',
      vehicleId: 'v1',
      title: 'Disobeying a Traffic Signal',
      description:
        'Vehicle 1 proceeded through the intersection after the signal controlling its lane had turned red.',
      lawSection: 'SAMPLE-VTL-1111',
      severity: 'major',
      evidence: [
        {
          id: 'ev-001',
          type: 'video_timestamp',
          description: 'Signal turns red; Vehicle 1 has not yet stopped',
          timestamp: 8.9,
        },
        {
          id: 'ev-002',
          type: 'video_timestamp',
          description: 'Vehicle 1 crosses stop line under red indication',
          timestamp: 9.1,
        },
      ],
    },
    {
      id: 'viol-002',
      vehicleId: 'v1',
      title: 'Failure to Yield Right of Way',
      description:
        'By entering the intersection against the signal, Vehicle 1 failed to yield right of way to Vehicle 2 which was lawfully proceeding on green.',
      lawSection: 'SAMPLE-VTL-1140',
      severity: 'major',
      evidence: [
        {
          id: 'ev-003',
          type: 'law_citation',
          description: 'Right of way at intersection controlled by traffic signals',
          lawId: 'law-right-of-way',
        },
      ],
    },
  ],
  conflictingEvidence: [
    'Exact moment Vehicle 1 driver perceived the signal change cannot be determined from footage.',
    'The camera angle does not allow a direct view of the Vehicle 1 driver\'s sight line toward the traffic signal.',
  ],
  missingInfo: [
    'Vehicle telemetry / EDR data for both vehicles',
    'Witness statements from bystanders on the northwest corner',
    'Signal phase timing logs from NYC DOT for this intersection',
    'Medical assessment of driver condition (distraction, impairment)',
  ],
  confidenceLevel: 0.78,
  limitations: [
    'This analysis is AI-assisted and based solely on the provided video footage. It is not a legal determination.',
    'Speed estimates are approximations derived from visual observation.',
    'Signal timing interpretation depends on camera angle and may not capture all relevant context.',
    'This report does not constitute legal advice or an insurance decision.',
  ],
};

// ---------------------------------------------------------------------------
// Vehicle Tracks — ACC-001
// ---------------------------------------------------------------------------
const tracks001: VehicleTrack[] = [
  {
    id: 'v1',
    label: 'Vehicle 1',
    class: 'Sedan',
    color: '#3b82f6',
    speed: 52,
    firstSeen: 1.0,
    lastSeen: 12.5,
    confidence: 0.91,
    events: [
      'Approaching intersection at t=2.1s',
      'Yellow signal at t=5.4s — no braking observed',
      'Red signal at t=8.9s — crosses stop line at t=9.1s',
      'Impact with Vehicle 2 at t=10.3s',
      'Vehicle comes to rest at t=12.5s',
    ],
    frames: makeFrames('v1', 1.0, 12.5, 0.45, 0.05, 0.0, 0.055, 28),
  },
  {
    id: 'v2',
    label: 'Vehicle 2',
    class: 'SUV',
    color: '#ef4444',
    speed: 38,
    firstSeen: 6.5,
    lastSeen: 14.0,
    confidence: 0.88,
    events: [
      'Enters frame from west at t=6.5s',
      'Approaching intersection on green signal at t=9.2s',
      'Impact at t=10.3s',
      'Post-impact rotation observed at t=10.3–11.8s',
      'Vehicle comes to rest at t=14.0s',
    ],
    frames: makeFrames('v2', 6.5, 14.0, 0.05, 0.42, 0.065, 0.0, 28),
  },
];

// ---------------------------------------------------------------------------
// VLM Analysis — ACC-003 (partial — intersection obstruction)
// ---------------------------------------------------------------------------
const vlmAnalysis003: VLMAnalysis = {
  summary:
    'A rear-end collision captured on a highway on-ramp. Vehicle 1 (white pickup truck) is traveling at highway speed when Vehicle 2 (blue hatchback) decelerates sharply ahead of it. Vehicle 1 fails to maintain adequate following distance and rear-ends Vehicle 2.',
  timeline: [
    {
      timestamp: 1.5,
      type: 'fact',
      content: 'Vehicle 2 (blue hatchback) is visible in the right lane of an on-ramp, merging into highway traffic.',
    },
    {
      timestamp: 3.0,
      type: 'fact',
      content: 'Vehicle 2 decelerates sharply — brake lights activate.',
    },
    {
      timestamp: 3.2,
      type: 'interpretation',
      content: 'Vehicle 2 appears to be reacting to a gap in merging traffic; exact reason for deceleration is unclear.',
    },
    {
      timestamp: 3.8,
      type: 'fact',
      content: 'Vehicle 1 (white pickup) brake lights activate — approximately 0.8 seconds after Vehicle 2 begins braking.',
    },
    {
      timestamp: 5.1,
      type: 'fact',
      content: 'Vehicle 1 rear-ends Vehicle 2. Impact speed estimated visually at 30–40 km/h relative velocity.',
    },
    {
      timestamp: 6.5,
      type: 'fact',
      content: 'Vehicle 2 is pushed forward approximately 4 meters. Both vehicles decelerate to a stop within 8 meters.',
    },
  ],
  vehicleActions: {
    'v1': 'Was following Vehicle 2 at insufficient distance; reacted approximately 0.8s after Vehicle 2 braked; was unable to stop in time.',
    'v2': 'Decelerated sharply while merging; brake lights functioned correctly.',
  },
  collisionDescription:
    'A rear-end impact at approximately 30–40 km/h relative velocity on a highway on-ramp. Vehicle 1 struck the rear bumper and trunk of Vehicle 2 head-on.',
  trafficSignals: 'No traffic signals visible in the footage. Area is an uncontrolled highway on-ramp.',
  roadContext:
    'Highway on-ramp, single lane transitioning to two lanes. Dry road surface, daytime, clear visibility.',
  uncertainties: [
    'Exact cause of Vehicle 2\'s sharp deceleration is not visible in the footage.',
    'Following distance at the start of the clip cannot be precisely measured without calibration references.',
  ],
};

const tracks003: VehicleTrack[] = [
  {
    id: 'v1',
    label: 'Vehicle 1',
    class: 'Pickup Truck',
    color: '#f59e0b',
    speed: 68,
    firstSeen: 0.5,
    lastSeen: 8.0,
    confidence: 0.85,
    events: [
      'Following Vehicle 2 on on-ramp at t=0.5s',
      'Brake lights activate at t=3.8s',
      'Rear-end impact at t=5.1s',
    ],
    frames: makeFrames('v1', 0.5, 8.0, 0.38, 0.60, 0.008, -0.01, 25),
  },
  {
    id: 'v2',
    label: 'Vehicle 2',
    class: 'Hatchback',
    color: '#06b6d4',
    speed: 32,
    firstSeen: 0.2,
    lastSeen: 8.0,
    confidence: 0.87,
    events: [
      'Merging on on-ramp at t=1.5s',
      'Sharp deceleration / brake lights at t=3.0s',
      'Impact received at t=5.1s',
    ],
    frames: makeFrames('v2', 0.2, 8.0, 0.38, 0.45, 0.008, -0.008, 25),
  },
];

// ---------------------------------------------------------------------------
// MOCK_ACCIDENTS — 6 records with varied statuses
// ---------------------------------------------------------------------------
export const MOCK_ACCIDENTS: AccidentRecord[] = [
  {
    id: 'ACC-001',
    videoFilename: 'intersection_collision_downtown.mp4',
    uploadDate: '2025-01-14T09:32:00Z',
    location: 'Intersection of 5th Ave & W 42nd St, Manhattan, NY',
    incidentDate: '2025-01-13T17:45:00Z',
    description:
      'Red-light running collision at a signalized intersection during evening rush hour.',
    notes: 'Two vehicles involved. Surveillance footage from a nearby business camera. (Demo Data)',
    status: 'completed',
    vehicleCount: 2,
    liabilityStatus: 'assessed',
    tracks: tracks001,
    vlmAnalysis: vlmAnalysis001,
    liabilityAssessment: liabilityAssessment001,
    isDemoData: true,
  },
  {
    id: 'ACC-002',
    videoFilename: 'sideswipe_brooklyn_bridge_blvd.mp4',
    uploadDate: '2025-01-12T14:18:00Z',
    location: 'Brooklyn Bridge Blvd & Atlantic Ave, Brooklyn, NY',
    incidentDate: '2025-01-12T11:20:00Z',
    description: 'Lane-change sideswipe involving a sedan and a delivery van.',
    notes:
      'Dashcam footage from a third vehicle. Audio available but not analyzed. (Demo Data)',
    status: 'reasoning',
    vehicleCount: 2,
    liabilityStatus: 'pending',
    isDemoData: true,
  },
  {
    id: 'ACC-003',
    videoFilename: 'highway_rear_end_onramp.mp4',
    uploadDate: '2025-01-10T08:05:00Z',
    location: 'I-278 East on-ramp at BQE, Queens, NY',
    incidentDate: '2025-01-09T14:12:00Z',
    description: 'Rear-end collision on a highway on-ramp during lane merge.',
    notes: 'Traffic camera footage sourced from NYC DOT. (Demo Data)',
    status: 'completed',
    vehicleCount: 2,
    liabilityStatus: 'assessed',
    tracks: tracks003,
    vlmAnalysis: vlmAnalysis003,
    liabilityAssessment: {
      overallAssessment:
        'Evidence suggests Vehicle 1 (white pickup) bears primary responsibility for maintaining insufficient following distance. Vehicle 2\'s deceleration, while sharp, was within normal driving behavior.',
      overallLevel: 'potential_responsibility',
      vehicleAnalyses: [
        {
          vehicleId: 'v1',
          vehicleLabel: 'Vehicle 1 — White Pickup Truck',
          liabilityLevel: 'potential_responsibility',
          summary:
            'Insufficient following distance and delayed braking response are the primary contributing factors.',
          contributingActions: ['Insufficient following distance', 'Delayed braking response'],
          mitigatingFactors: ['Road merge situation may have reduced visibility ahead'],
          evidence: [
            {
              id: 'ev-r-001',
              type: 'video_timestamp',
              description: 'Vehicle 1 brake lights activate 0.8s after Vehicle 2 begins braking',
              timestamp: 3.8,
            },
          ],
        },
        {
          vehicleId: 'v2',
          vehicleLabel: 'Vehicle 2 — Blue Hatchback',
          liabilityLevel: 'insufficient_evidence',
          summary: 'Vehicle 2 operated its braking system correctly. Cause of deceleration is unclear but not attributable to reckless behavior.',
          contributingActions: [],
          mitigatingFactors: ['Brake lights functioned and activated promptly'],
          evidence: [],
        },
      ],
      violations: [
        {
          id: 'viol-r-001',
          vehicleId: 'v1',
          title: 'Unsafe Following Distance',
          description: 'Vehicle 1 failed to maintain a safe following distance as required.',
          lawSection: 'SAMPLE-VTL-1129',
          severity: 'moderate',
          evidence: [
            {
              id: 'ev-r-002',
              type: 'video_timestamp',
              description: 'Insufficient gap visible between vehicles before braking event',
              timestamp: 2.0,
            },
          ],
        },
      ],
      conflictingEvidence: [
        'The abruptness of Vehicle 2\'s deceleration may partially excuse Vehicle 1\'s inability to stop in time.',
      ],
      missingInfo: [
        'Exact following distance at incident start',
        'Speed readings from both vehicles',
        'Weather/road condition data',
      ],
      confidenceLevel: 0.71,
      limitations: [
        'AI-assisted analysis — not a legal determination.',
        'Speed and distance estimates are visual approximations only.',
      ],
    },
    isDemoData: true,
  },
  {
    id: 'ACC-004',
    videoFilename: 'parking_lot_fender_bender.mp4',
    uploadDate: '2025-01-08T16:45:00Z',
    location: 'Costco Parking Lot, College Point, Queens, NY',
    incidentDate: '2025-01-08T15:30:00Z',
    description: 'Low-speed parking lot collision between two SUVs backing out simultaneously.',
    notes: 'Private parking lot CCTV. No traffic signals involved. (Demo Data)',
    status: 'vlm_analysis',
    vehicleCount: 2,
    liabilityStatus: 'pending',
    isDemoData: true,
  },
  {
    id: 'ACC-005',
    videoFilename: 'pedestrian_near_miss_crosswalk.mp4',
    uploadDate: '2025-01-06T11:22:00Z',
    location: 'W 34th St & 7th Ave, Midtown, NY',
    incidentDate: '2025-01-05T09:15:00Z',
    description:
      'Vehicle failed to yield to pedestrian in marked crosswalk during pedestrian signal phase.',
    notes:
      'Incident did not result in contact. Included as near-miss for review. (Demo Data)',
    status: 'tracking',
    vehicleCount: 1,
    liabilityStatus: 'pending',
    isDemoData: true,
  },
  {
    id: 'ACC-006',
    videoFilename: 'damaged_upload_corrupted.mp4',
    uploadDate: '2025-01-04T08:10:00Z',
    location: 'Bronx River Pkwy, Bronx, NY',
    incidentDate: '2025-01-03T22:40:00Z',
    description: 'Multi-vehicle incident on expressway. Video file was corrupted during upload.',
    notes: 'Analysis failed due to unreadable video stream. Re-upload required. (Demo Data)',
    status: 'failed',
    vehicleCount: 3,
    liabilityStatus: 'pending',
    isDemoData: true,
  },
];

export { vlmAnalysis001, liabilityAssessment001, tracks001 };
