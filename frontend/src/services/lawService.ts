// Traffic law service — mock mode or real FastAPI backend
// DEMO DATA — replace MOCK_* imports with real API calls when backend is connected

import type { TrafficLaw, LawSearchResult } from '../types/law';
import { MOCK_TRAFFIC_LAWS } from '../mocks/lawMocks';
import { isMockMode, mockDelay, get } from './apiClient';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LawSearchFilters {
  jurisdiction?: string;
  topic?: string;
  section?: string;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Search traffic laws by query string, with optional filters */
export async function searchLaws(
  query: string = '',
  filters: LawSearchFilters = {},
): Promise<LawSearchResult[]> {
  if (isMockMode) {
    await mockDelay(300);
    const lower = query.toLowerCase();

    const filtered = MOCK_TRAFFIC_LAWS.filter((law) => {
      // Text match
      const matchesQuery =
        !query ||
        law.title.toLowerCase().includes(lower) ||
        law.section.toLowerCase().includes(lower) ||
        law.excerpt.toLowerCase().includes(lower) ||
        law.fullText.toLowerCase().includes(lower);

      // Jurisdiction filter
      const matchesJurisdiction =
        !filters.jurisdiction ||
        law.jurisdiction.toLowerCase().includes(filters.jurisdiction.toLowerCase());

      // Topic filter
      const matchesTopic =
        !filters.topic ||
        law.topics.some((t) => t.toLowerCase().includes(filters.topic!.toLowerCase()));

      // Section filter
      const matchesSection =
        !filters.section ||
        law.section.toLowerCase().includes(filters.section.toLowerCase());

      return matchesQuery && matchesJurisdiction && matchesTopic && matchesSection;
    });

    return filtered.map((law) => ({
      law,
      matchScore: law.relevanceScore ?? 0.5,
      matchedFields: query
        ? (['title', 'section', 'excerpt', 'fullText'] as const).filter((field) =>
            (law[field] as string).toLowerCase().includes(lower),
          )
        : [],
    }));
  }

  return get<LawSearchResult[]>('/api/laws/search', {
    q: query,
    ...filters,
  });
}

/** Fetch a single traffic law by ID */
export async function getLaw(id: string): Promise<TrafficLaw> {
  if (isMockMode) {
    await mockDelay(300);
    const law = MOCK_TRAFFIC_LAWS.find((l) => l.id === id);
    if (!law) throw new Error(`Law ${id} not found`);
    return law;
  }
  return get<TrafficLaw>(`/api/laws/${id}`);
}
