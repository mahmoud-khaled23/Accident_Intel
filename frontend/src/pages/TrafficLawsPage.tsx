import { useState, useEffect, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import { PageHeader } from '@/components/common/PageHeader';
import { DemoBadge } from '@/components/common/DemoBadge';
import { EmptyState } from '@/components/common/EmptyState';
import { LoadingState } from '@/components/common/LoadingState';
import { LegalCitationCard } from '@/components/analysis/LegalCitationCard';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { searchLaws } from '@/services/lawService';
import { MOCK_TRAFFIC_LAWS } from '@/mocks/lawMocks';
import type { LawSearchResult } from '@/types/law';
import { cn } from '@/lib/utils';

// Derive filter chip values from mock data (computed once at module level)
const ALL_JURISDICTIONS = [...new Set(MOCK_TRAFFIC_LAWS.map((l) => l.jurisdiction))];
const ALL_TOPICS = [...new Set(MOCK_TRAFFIC_LAWS.flatMap((l) => l.topics))];

export function TrafficLawsPage() {
  const [query, setQuery] = useState('');
  const [jurisdictionFilter, setJurisdictionFilter] = useState<string | null>(null);
  const [topicFilter, setTopicFilter] = useState<string | null>(null);
  const [results, setResults] = useState<LawSearchResult[]>([]);
  const [loading, setLoading] = useState(true);

  const runSearch = useCallback(async (q: string, jur: string | null, topic: string | null) => {
    setLoading(true);
    try {
      const found = await searchLaws(q, {
        jurisdiction: jur ?? undefined,
        topic: topic ?? undefined,
      });
      setResults(found);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced search on query/filter change
  useEffect(() => {
    const id = setTimeout(() => {
      void runSearch(query, jurisdictionFilter, topicFilter);
    }, 300);
    return () => clearTimeout(id);
  }, [query, jurisdictionFilter, topicFilter, runSearch]);

  function clearAll() {
    setQuery('');
    setJurisdictionFilter(null);
    setTopicFilter(null);
  }

  const hasFilters = query || jurisdictionFilter || topicFilter;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Traffic Laws Library"
        description="Searchable NYC traffic law reference"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Traffic Laws' },
        ]}
        actions={<DemoBadge />}
      />

      {/* Search input */}
      <div className="relative max-w-xl">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500 pointer-events-none" />
        <Input
          placeholder="Search by title, section, or content…"
          className="pl-9 pr-10"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search traffic laws"
        />
        {query && (
          <button
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
            onClick={() => setQuery('')}
            aria-label="Clear search"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Jurisdiction filter chips */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Jurisdiction</p>
        <div className="flex flex-wrap gap-2">
          {ALL_JURISDICTIONS.map((jur) => (
            <button
              key={jur}
              onClick={() =>
                setJurisdictionFilter((prev) => (prev === jur ? null : jur))
              }
              className={cn(
                'rounded-full px-3 py-1 text-xs font-medium transition-colors',
                jurisdictionFilter === jur
                  ? 'bg-primary text-white'
                  : 'bg-[#1e2d45] text-slate-400 hover:bg-[#253348]',
              )}
            >
              {jur}
            </button>
          ))}
        </div>
      </div>

      {/* Topic filter chips */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Topic</p>
        <div className="flex flex-wrap gap-2">
          {ALL_TOPICS.map((topic) => (
            <button
              key={topic}
              onClick={() =>
                setTopicFilter((prev) => (prev === topic ? null : topic))
              }
              className={cn(
                'rounded-full px-3 py-1 text-xs font-medium transition-colors capitalize',
                topicFilter === topic
                  ? 'bg-primary text-white'
                  : 'bg-[#1e2d45] text-slate-400 hover:bg-[#253348]',
              )}
            >
              {topic.replace(/-/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Results count + clear */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {loading ? 'Searching…' : `${results.length} result${results.length !== 1 ? 's' : ''}`}
        </p>
        {hasFilters && (
          <Button variant="ghost" size="sm" className="text-xs" onClick={clearAll}>
            <X className="h-3 w-3 mr-1" /> Clear all filters
          </Button>
        )}
      </div>

      {/* Results */}
      {loading ? (
        <LoadingState message="Searching laws…" />
      ) : results.length === 0 ? (
        <EmptyState
          title="No laws found"
          description="Try adjusting your search or clearing the filters."
          action={
            hasFilters ? (
              <Button variant="outline" size="sm" onClick={clearAll}>
                Clear filters
              </Button>
            ) : undefined
          }
        />
      ) : (
        <div className="space-y-4">
          {results.map((result) => (
            <LegalCitationCard key={result.law.id} law={result.law} />
          ))}
        </div>
      )}

      {/* Disclaimer */}
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-300">
        <strong>Disclaimer:</strong> All law entries are sample demo text created solely for UI
        demonstration. Section identifiers are prefixed with "SAMPLE-" and do not represent real
        official law text. Do not use this library as legal reference.
      </div>
    </div>
  );
}
