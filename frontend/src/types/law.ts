// Legal-domain type definitions for Accident-Intel

/** A traffic law or regulation entry */
export interface TrafficLaw {
  id: string;
  title: string;
  jurisdiction: string; // e.g. "NYC", "New York State"
  section: string; // e.g. "VTL-1140"
  excerpt: string; // short quotable passage
  fullText: string; // complete text of the provision
  whyApplies?: string; // AI-generated explanation of relevance to the accident
  sourceUrl?: string; // official publication URL (undefined = not available)
  effectiveDate?: string; // ISO 8601 date string
  relevanceScore?: number; // 0–1, returned by RAG retrieval
  topics: string[]; // e.g. ["right-of-way", "intersection"]
  article?: string; // VTL article number, e.g. "24"
  category?: string; // corpus category, e.g. "core", "enforcement_program"
  isDemoData?: boolean;
}

/** A law section sub-unit (for laws with multiple subsections) */
export interface LawSection {
  subsection: string; // e.g. "(a)", "(1)"
  text: string;
}

/** A law retrieved during RAG for a specific accident analysis */
export interface RetrievedLaw extends TrafficLaw {
  analysisId: string;
  retrievalRank: number; // 1-indexed rank in retrieval results
}

/** Search result wrapping a TrafficLaw with match context */
export interface LawSearchResult {
  law: TrafficLaw;
  matchScore: number; // 0–1
  matchedFields: string[]; // which fields matched the query
}

/** Facet counts over the full law corpus (browse responses) */
export interface LawFacets {
  categories: Record<string, number>;
  articles: Record<string, number>;
}

/** One page of browse/search results from the laws API */
export interface LawSearchPage {
  results: LawSearchResult[];
  total: number; // total matches before pagination slicing
  facets?: LawFacets;
}
