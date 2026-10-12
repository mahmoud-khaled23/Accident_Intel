// DEMO DATA: These are fictional law samples for UI prototyping. Do not use as legal reference.
// DEMO DATA — replace with real API responses when backend is connected
//
// All section identifiers are prefixed with 'SAMPLE-' to clearly indicate they are
// fictional placeholders created solely for UI demonstration. They do NOT represent
// real law text, official citations, or legal advice.

import type { TrafficLaw } from '../types/law';

export const MOCK_TRAFFIC_LAWS: TrafficLaw[] = [
  // ------------------------------------------------------------------
  // 1. Right of Way at Intersections
  // ------------------------------------------------------------------
  {
    id: 'law-right-of-way',
    title: 'Right of Way at Intersections',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1140',
    excerpt:
      'The driver of a vehicle approaching an intersection shall yield the right of way to a vehicle which has entered the intersection from a different highway.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1140. Vehicle approaching or entering intersection.\n\n(a) The driver of a vehicle approaching an intersection shall yield the right of way to a vehicle which has entered the intersection from a different highway.\n\n(b) When two vehicles enter an intersection from different highways at the same time, the driver of the vehicle on the left shall yield the right of way to the vehicle on the right.\n\n(c) Nothing in this section shall affect the right of a driver to proceed through a traffic control signal indicating the right to proceed.',
    whyApplies:
      'Relevant when one vehicle enters an intersection while another vehicle already has right of way, particularly in cases involving conflicting signal indications or uncontrolled intersections.',
    sourceUrl: 'https://example.gov/sample-vtl-1140',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.92,
    topics: ['right-of-way', 'intersection'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 2. Obedience to Traffic Signals
  // ------------------------------------------------------------------
  {
    id: 'law-signals',
    title: 'Traffic Control Signals — Obedience Requirements',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1111',
    excerpt:
      'Whenever traffic is controlled by traffic control signals exhibiting different colored lights, the following colors shall indicate and apply to drivers of vehicles: Green — vehicular traffic facing the signal may proceed; Yellow — vehicular traffic facing the signal is warned that the red signal is imminent; Red — vehicular traffic facing the signal shall stop.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1111. Traffic control signals.\n\n(a) Whenever traffic is controlled by traffic control signals exhibiting the words "Go," "Caution," or "Stop," or exhibiting different colored lights successively one at a time, or with arrows, the following colors only shall be used and their meaning shall apply to drivers of vehicles and pedestrians as follows:\n\n1. Green indication:\n   (a) Vehicular traffic facing a circular green signal may proceed straight through or turn right or left unless a sign at such place prohibits either such turn. But vehicular traffic, including vehicles turning right or left, shall yield the right-of-way to other vehicles and to pedestrians lawfully within the intersection or an adjacent crosswalk at the time such signal is exhibited.\n\n2. Yellow indication:\n   (a) Vehicular traffic facing a steady yellow signal is thereby warned that the related green movement is being terminated or that a red indication will be exhibited immediately thereafter.\n\n3. Red indication:\n   (a) Vehicular traffic facing a steady red signal alone shall stop before entering the crosswalk on the near side of the intersection or, if none, then before entering the intersection and shall remain standing until an indication to proceed is shown.',
    whyApplies:
      'Directly applicable when a vehicle fails to obey red or yellow signal indications at an intersection, which is a primary contributing factor in signal-violation collisions.',
    sourceUrl: 'https://example.gov/sample-vtl-1111',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.97,
    topics: ['signals', 'intersection', 'red-light'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 3. Speed Limits
  // ------------------------------------------------------------------
  {
    id: 'law-speed',
    title: 'Maximum Speed Limits',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1180',
    excerpt:
      'No person shall drive a vehicle at a speed greater than is reasonable and prudent under the conditions and having regard to the actual and potential hazards then existing.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1180. Speed restrictions.\n\n(a) No person shall drive a vehicle at a speed greater than is reasonable and prudent under the conditions and having regard to the actual and potential hazards then existing.\n\n(b) The maximum speed limits are:\n   1. Twenty-five miles per hour in a city or village.\n   2. Thirty miles per hour on a roadway immediately outside of a city or village.\n   3. Fifty-five miles per hour on all other public highways.\n\n(c) The fact that the speed of a vehicle is lower than the maximum speed limits shall not relieve the driver from the duty to decrease speed when approaching and crossing an intersection or a railway grade crossing, when approaching and going around a curve, when approaching a hillcrest, when traveling upon any narrow or winding roadway, or when special hazard exists.',
    whyApplies:
      'Applies when pre-impact vehicle speed is assessed as contributing to the severity or occurrence of the collision, including at intersections and on highways.',
    sourceUrl: 'https://example.gov/sample-vtl-1180',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.74,
    topics: ['speed', 'intersection', 'highway'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 4. Lane Changes
  // ------------------------------------------------------------------
  {
    id: 'law-lane-change',
    title: 'Driving on Roadways Laned for Traffic — Lane Changes',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1128',
    excerpt:
      'A vehicle shall be driven as nearly as practicable entirely within a single lane and shall not be moved from such lane until the driver has first ascertained that such movement can be made with safety.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1128. Driving on roadways laned for traffic.\n\nWhenever any roadway has been divided into two or more clearly marked lanes for traffic the following rules in addition to all others consistent herewith shall apply:\n\n(a) A vehicle shall be driven as nearly as practicable entirely within a single lane and shall not be moved from such lane until the driver has first ascertained that such movement can be made with safety.\n\n(b) Upon a roadway which is divided into three lanes and provides for two-way movement of traffic, a vehicle shall not be driven in the center lane except when overtaking and passing another vehicle traveling in the same direction when such center lane is clear of traffic within a safe distance, or in preparation for making a left turn or where such center lane is at the time allocated exclusively to traffic moving in the same direction that the vehicle is proceeding.\n\n(c) Official traffic control devices may be erected directing specified traffic to use a designated lane or designating those lanes to be used by traffic moving in a particular direction regardless of the center of the roadway.',
    whyApplies:
      'Relevant in sideswipe collisions and incidents where a vehicle changes lanes without confirming the movement is safe, including merging incidents.',
    sourceUrl: 'https://example.gov/sample-vtl-1128',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.85,
    topics: ['lane-changes', 'merging'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 5. Following Distance
  // ------------------------------------------------------------------
  {
    id: 'law-following-distance',
    title: 'Following Too Closely',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1129',
    excerpt:
      'The driver of a motor vehicle shall not follow another vehicle more closely than is reasonable and prudent, having due regard for the speed of such vehicles and the traffic upon and the condition of the highway.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1129. Following too closely.\n\n(a) The driver of a motor vehicle shall not follow another vehicle more closely than is reasonable and prudent, having due regard for the speed of such vehicles and the traffic upon and the condition of the highway.\n\n(b) The driver of any motor truck or motor vehicle drawing another vehicle when traveling upon a roadway outside of a business or residence district and which is following another motor truck or motor vehicle drawing another vehicle shall, whenever conditions permit, leave sufficient space so that an overtaking vehicle may enter and occupy such space without danger, except that this shall not prevent a motor truck or motor vehicle drawing another vehicle from overtaking and passing any like vehicle or other vehicle.',
    whyApplies:
      'Central to rear-end collision analysis; establishes the legal standard for safe following distance and driver responsibility to avoid collisions with vehicles ahead.',
    sourceUrl: 'https://example.gov/sample-vtl-1129',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.95,
    topics: ['following-distance', 'rear-end'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 6. Turn Signals
  // ------------------------------------------------------------------
  {
    id: 'law-turn-signals',
    title: 'Signal Requirements Before Turning or Changing Lanes',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1163',
    excerpt:
      'No person shall turn a vehicle at an intersection unless the vehicle is in proper position upon the roadway, or turn a vehicle to enter a private road or driveway, or otherwise turn a vehicle from a direct course or move right or left upon a roadway unless and until such movement can be made with reasonable safety.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1163. Turning movements and required signals.\n\n(a) No person shall turn a vehicle at an intersection unless the vehicle is in proper position upon the roadway as required in sections 1160 and 1161, or turn a vehicle to enter a private road or driveway, or otherwise turn a vehicle from a direct course or move right or left upon a roadway unless and until such movement can be made with reasonable safety. No person shall so turn any vehicle without giving an appropriate signal in the manner hereinafter provided.\n\n(b) A signal of intention to turn or move right or left when required shall be given continuously during not less than the last one hundred feet traveled by the vehicle before turning.\n\n(c) A vehicle must not be stopped or suddenly decreased in speed without first giving an appropriate signal in the manner provided herein to the driver of any vehicle immediately to the rear when there is opportunity to give such signal.',
    whyApplies:
      'Applicable when a vehicle turns or changes lanes without signaling, creating a hazard for other drivers who could not anticipate the movement.',
    sourceUrl: 'https://example.gov/sample-vtl-1163',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.68,
    topics: ['turn-signals', 'turning', 'lane-changes'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 7. Pedestrian Right of Way in Crosswalk
  // ------------------------------------------------------------------
  {
    id: 'law-pedestrian-crosswalk',
    title: 'Pedestrians\' Right of Way in Crosswalks',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1151',
    excerpt:
      'When traffic control signals are not in place or not in operation the driver of a vehicle shall yield the right of way, slowing down or stopping if need be, to a pedestrian crossing the roadway within a crosswalk.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1151. Pedestrians\' right of way in crosswalks.\n\n(a) When traffic control signals are not in place or not in operation the driver of a vehicle shall yield the right of way, slowing down or stopping if need be so as to yield to a pedestrian crossing the roadway within a crosswalk when the pedestrian is upon the half of the roadway upon which the vehicle is traveling, or when the pedestrian is approaching so closely from the opposite half of the roadway as to be in danger.\n\n(b) No pedestrian shall suddenly leave a curb or other place of safety and walk or run into the path of a vehicle which is so close that it is impractical for the driver to yield.\n\n(c) Whenever any vehicle is stopped at a marked crosswalk or at any unmarked crosswalk at an intersection to permit a pedestrian to cross the roadway, the driver of any other vehicle approaching from the rear shall not overtake and pass such stopped vehicle.',
    whyApplies:
      'Relevant in incidents where a driver fails to yield to a pedestrian in a marked or unmarked crosswalk, including near-miss events at signalized intersections.',
    sourceUrl: 'https://example.gov/sample-vtl-1151',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.88,
    topics: ['pedestrians', 'crosswalk', 'right-of-way'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 8. Reckless Driving
  // ------------------------------------------------------------------
  {
    id: 'law-reckless-driving',
    title: 'Reckless Driving',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1212',
    excerpt:
      'Reckless driving shall mean driving or using any motor vehicle in a manner which unreasonably interferes with the free and proper use of the public highway, or unreasonably endangers users of the public highway.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1212. Reckless driving.\n\nReckless driving shall mean driving or using any motor vehicle, motorcycle or any other vehicle propelled by any power other than muscular power or any appliance or accessory thereof in a manner which unreasonably interferes with the free and proper use of the public highway, or unreasonably endangers users of the public highway. Reckless driving is prohibited.',
    whyApplies:
      'May apply in cases where driving behavior is characterized by willful disregard for traffic safety rules, such as excessive speed in a school zone or repeated signal violations.',
    sourceUrl: 'https://example.gov/sample-vtl-1212',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.58,
    topics: ['reckless-driving', 'safety'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 9. Stopping and Yielding
  // ------------------------------------------------------------------
  {
    id: 'law-stop-sign',
    title: 'Obedience to Stop Signs',
    jurisdiction: 'NYC / New York State',
    section: 'SAMPLE-VTL-1172',
    excerpt:
      'Every driver of a vehicle approaching a stop sign shall stop at a clearly marked stop line, but if none, then shall stop before entering the crosswalk on the near side of the intersection.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1172. Stop signs and yield signs.\n\n(a) Except when directed to proceed by a police officer or traffic control signal, every driver of a vehicle approaching a stop sign shall stop at a clearly marked stop line, but if none, then shall stop before entering the crosswalk on the near side of the intersection, or if none, then at the point nearest the intersecting roadway where the driver has a view of the approaching traffic on the intersecting roadway before entering the intersection and the right to proceed shall be subject to the provisions of section 1142.\n\n(b) The driver of a vehicle approaching a yield sign shall in obedience to such sign slow down to a speed reasonable for the existing conditions and, if required for safety to stop, shall stop at a clearly marked stop line.',
    whyApplies:
      'Directly applicable when a driver fails to obey a stop sign before entering an intersection, a common factor in T-bone and angle collisions at uncontrolled intersections.',
    sourceUrl: 'https://example.gov/sample-vtl-1172',
    effectiveDate: '2000-01-01',
    relevanceScore: 0.76,
    topics: ['stop-signs', 'yielding', 'intersection'],
    isDemoData: true,
  },

  // ------------------------------------------------------------------
  // 10. School Zone Speed
  // ------------------------------------------------------------------
  {
    id: 'law-school-zone',
    title: 'Speed Limits in School Zones',
    jurisdiction: 'NYC',
    section: 'SAMPLE-VTL-1174',
    excerpt:
      'No driver of a vehicle shall pass another vehicle stopped at a crosswalk adjacent to a school building between the hours specified on an official sign.',
    fullText:
      '[SAMPLE DEMO TEXT — NOT REAL LAW]\n\nSAMPLE-VTL § 1174. Overtaking and passing school bus.\n\n(a) The driver of a vehicle upon a public highway, street or private road upon meeting or overtaking from either direction any school bus marked as such which has stopped for the purpose of receiving or discharging any pupil, shall stop the vehicle before reaching such school bus when there is in operation on said school bus a red visual signal as specified in subdivision twenty-six of section three hundred seventy-five of this chapter and said driver shall not proceed until such school bus resumes motion, or until said red visual signal is no longer actuated.\n\n[SAMPLE NYC AMENDMENT]\n\nWithin the City of New York, the posted speed limit in a designated school speed zone during school hours is 15 miles per hour. Automated speed enforcement cameras may be used in these zones.',
    whyApplies:
      'Relevant when a collision or near-miss occurs near a school building during school hours, where enhanced speed restrictions and pedestrian protections apply.',
    sourceUrl: 'https://example.gov/sample-vtl-1174',
    effectiveDate: '2014-07-01',
    relevanceScore: 0.45,
    topics: ['speed', 'school-zone', 'pedestrians'],
    isDemoData: true,
  },
];
