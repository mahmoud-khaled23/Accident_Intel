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

  // Real mode — call the backend search endpoint
  interface BackendLawResult {
    section_id: string;
    title: string;
    content: string;
    score: number;
    metadata: Record<string, unknown>;
  }
  interface BackendLawSearchResponse {
    results: BackendLawResult[];
    query: string;
    total: number;
  }

  const response = await get<BackendLawSearchResponse>('/api/laws/', {
    q: query || ' ', // send a space so backend returns mock laws when empty
    jurisdiction: filters.jurisdiction,
    topic: filters.topic,
    limit: 20,
  });

  return response.results.map((item) => ({
    law: {
      id: item.section_id,
      section: item.section_id,
      sectionId: item.section_id,
      title: item.title,
      jurisdiction: (item.metadata?.jurisdiction as string) ?? 'NYS VTL',
      excerpt: item.content,
      fullText: item.content,
      sourceUrl: (item.metadata?.source_url as string) || undefined,
      topics: [],
      relevanceScore: item.score,
    } as TrafficLaw,
    matchScore: item.score,
    matchedFields: [],
  }));
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
