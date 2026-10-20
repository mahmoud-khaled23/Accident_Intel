// Traffic law service — mock mode or real FastAPI backend.
// Real mode browses/searches the local NYS VTL corpus exposed by /api/laws.

import type { TrafficLaw, LawSearchResult, LawSearchPage } from '../types/law';
import { MOCK_TRAFFIC_LAWS } from '../mocks/lawMocks';
import { isMockMode, mockDelay, get } from './apiClient';

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

export interface LawSearchFilters {
  /** Exact corpus category, e.g. "core" or "enforcement_program" */
  category?: string;
  /** Exact VTL article number, e.g. "24" */
  article?: string;
}

export interface LawSearchOptions {
  /** Page size (default 20) */
  limit?: number;
  /** Number of matches to skip — for "Load more" pagination */
  offset?: number;
}

// ---------------------------------------------------------------------------
// Backend response shapes
// ---------------------------------------------------------------------------

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
  offset: number;
  facets?: { categories: Record<string, number>; articles: Record<string, number> };
}

// ---------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------

/** First paragraph of the law text, trimmed to a quotable excerpt. */
function excerptOf(text: string): string {
  const firstParagraph = text.split(/\n\s*\n/)[0] ?? text;
  const flat = firstParagraph.replace(/\s+/g, ' ').trim();
  return flat.length > 280 ? `${flat.slice(0, 277)}…` : flat;
}

/** Map one backend law record to the TrafficLaw view model. */
function mapBackendLaw(item: BackendLawResult): TrafficLaw {
  const meta = item.metadata ?? {};
  const article = meta.article !== undefined && meta.article !== null ? String(meta.article) : undefined;
  const category = typeof meta.category === 'string' ? meta.category : undefined;
  // Search responses carry the retrieval queries that surfaced the section.
  const matched = Array.isArray(meta.matched_queries) ? meta.matched_queries : [];
  return {
    id: item.section_id,
    section: item.section_id,
    title: item.title || `VTL § ${String(item.section_id).replace(/^VTL-/, '')}`,
    jurisdiction: 'New York State',
    excerpt: excerptOf(item.content),
    fullText: item.content,
    topics: matched
      .filter((q): q is string => typeof q === 'string' && !q.startsWith('(referenced'))
      .map((q) => q.replace(/^rule: /, '')),
    article,
    category,
    relevanceScore: item.score || undefined,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Browse (empty query) or search the traffic-law corpus.
 *
 * With an empty query the backend lists the corpus in article/section order
 * with facet counts; with a query it runs hybrid RAG retrieval ordered by
 * relevance.
 */
export async function searchLaws(
  query: string = '',
  filters: LawSearchFilters = {},
  options: LawSearchOptions = {},
): Promise<LawSearchPage> {
  const limit = options.limit ?? 20;
  const offset = options.offset ?? 0;

  if (isMockMode) {
    await mockDelay(300);
    const lower = query.toLowerCase();
    const filtered = MOCK_TRAFFIC_LAWS.filter(
      (law) =>
        !query ||
        law.title.toLowerCase().includes(lower) ||
        law.section.toLowerCase().includes(lower) ||
        law.excerpt.toLowerCase().includes(lower) ||
        law.fullText.toLowerCase().includes(lower),
    );
    const results: LawSearchResult[] = filtered.slice(offset, offset + limit).map((law) => ({
      law,
      matchScore: law.relevanceScore ?? 0.5,
      matchedFields: [],
    }));
    return { results, total: filtered.length };
  }

  const params: Record<string, string | number> = { limit, offset };
  if (query.trim()) params.q = query.trim();
  if (filters.category) params.category = filters.category;
  if (filters.article) params.article = filters.article;

  // Semantic search pays a one-off dense-model load on the first request
  // after an API restart, so allow more than the default 30 s for it.
  const config = query.trim() ? { timeout: 120_000 } : undefined;
  const response = await get<BackendLawSearchResponse>('/api/laws/', params, config);
  return {
    results: response.results.map((item) => ({
      law: mapBackendLaw(item),
      matchScore: item.score ?? 0,
      matchedFields: [],
    })),
    total: response.total,
    facets: response.facets,
  };
}

/** Fetch a single traffic law by ID */
export async function getLaw(id: string): Promise<TrafficLaw> {
  if (isMockMode) {
    await mockDelay(300);
    const law = MOCK_TRAFFIC_LAWS.find((l) => l.id === id);
    if (!law) throw new Error(`Law ${id} not found`);
    return law;
  }
  const result = await get<BackendLawResult>(`/api/laws/${id}`);
  return mapBackendLaw(result);
}
