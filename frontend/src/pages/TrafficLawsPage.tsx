import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { BookOpen, Search, Scale, X } from 'lucide-react';
import { PageHeader } from '@/components/common/PageHeader';
import { DemoBadge } from '@/components/common/DemoBadge';
import { EmptyState } from '@/components/common/EmptyState';
import { ErrorState } from '@/components/common/ErrorState';
import { LoadingState } from '@/components/common/LoadingState';
import { LegalCitationCard } from '@/components/analysis/LegalCitationCard';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { searchLaws } from '@/services/lawService';
import { isMockMode } from '@/services/apiClient';
import type { LawSearchResult, LawFacets, TrafficLaw } from '@/types/law';
import { cn } from '@/lib/utils';

const PAGE_SIZE = 24;

const CATEGORY_LABELS: Record<string, string> = {
  core: 'Core provisions',
  enforcement_program: 'Enforcement programs',
};

const categoryLabel = (category: string) =>
  CATEGORY_LABELS[category] ?? category.replace(/_/g, ' ');

interface FilterChipProps {
  label: string;
  count?: number;
  active: boolean;
  onClick: () => void;
}

function FilterChip({ label, count, active, onClick }: FilterChipProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors',
        active
          ? 'bg-primary text-white'
          : 'bg-[#1e2d45] text-slate-400 hover:bg-[#253348]',
      )}
    >
      {label}
      {count !== undefined && (
        <span className={active ? 'text-white/70' : 'text-slate-500'}>{count}</span>
      )}
    </button>
  );
}

export function TrafficLawsPage() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [articleFilter, setArticleFilter] = useState<string | null>(null);
  const [results, setResults] = useState<LawSearchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [facets, setFacets] = useState<LawFacets | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debounce the free-text query (semantic search runs per query)
  useEffect(() => {
    const id = setTimeout(() => setDebouncedQuery(query), 300);
    return () => clearTimeout(id);
  }, [query]);

  // Monotonic id so a slow earlier request can never clobber newer results
  const requestIdRef = useRef(0);

  const load = useCallback(
    async (offset: number) => {
      if (offset === 0) {
        setLoading(true);
        setError(null);
      } else {
        setLoadingMore(true);
      }
      const seq = offset === 0 ? ++requestIdRef.current : requestIdRef.current;
      try {
        const page = await searchLaws(
          debouncedQuery,
          { category: categoryFilter ?? undefined, article: articleFilter ?? undefined },
          { limit: PAGE_SIZE, offset },
        );
        if (seq !== requestIdRef.current) return; // a newer request superseded this one
        setResults((prev) => (offset === 0 ? page.results : [...prev, ...page.results]));
        setTotal(page.total);
        if (offset === 0) setFacets(page.facets);
      } catch (err) {
        if (seq !== requestIdRef.current) return;
        setError(err instanceof Error ? err.message : 'Failed to load traffic laws.');
        if (offset === 0) {
          setResults([]);
          setTotal(0);
        }
      } finally {
        if (seq === requestIdRef.current) {
          setLoading(false);
          setLoadingMore(false);
        }
      }
    },
    [debouncedQuery, categoryFilter, articleFilter],
  );

  // Reload from the top whenever the query or filters change
  useEffect(() => {
    void load(0);
  }, [load]);

  const isSearching = debouncedQuery.trim().length > 0;
  const hasFilters = Boolean(categoryFilter || articleFilter);

  function clearAll() {
    setQuery('');
    setDebouncedQuery('');
    setCategoryFilter(null);
    setArticleFilter(null);
  }

  // Browse mode: organize sections under article headings
  const articleGroups = useMemo(() => {
    if (isSearching) return [];
    const groups = new Map<string, TrafficLaw[]>();
    for (const result of results) {
      const key = result.law.article ?? 'Other';
      const list = groups.get(key);
      if (list) list.push(result.law);
      else groups.set(key, [result.law]);
    }
    return [...groups.entries()];
  }, [results, isSearching]);

  const categoryEntries = Object.entries(facets?.categories ?? {});
  const articleEntries = Object.entries(facets?.articles ?? {}).sort(
    ([a], [b]) => Number(a) - Number(b),
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Traffic Laws Library"
        description="Browse and search the New York Vehicle & Traffic Law corpus used by the accident-analysis pipeline"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Traffic Laws' },
        ]}
        actions={isMockMode ? <DemoBadge /> : undefined}
      />

      {/* Search input */}
      <div className="relative max-w-xl">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500 pointer-events-none" />
        <Input
          placeholder="Semantic search — e.g. “failure to yield at an intersection”…"
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

      {/* Facet filters — driven by real corpus counts from the API */}
      {categoryEntries.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Category</p>
          <div className="flex flex-wrap gap-2">
            {categoryEntries.map(([category, count]) => (
              <FilterChip
                key={category}
                label={categoryLabel(category)}
                count={count}
                active={categoryFilter === category}
                onClick={() =>
                  setCategoryFilter((prev) => (prev === category ? null : category))
                }
              />
            ))}
          </div>
        </div>
      )}
      {articleEntries.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Article</p>
          <div className="flex flex-wrap gap-2">
            {articleEntries.map(([article, count]) => (
              <FilterChip
                key={article}
                label={`Art. ${article}`}
                count={count}
                active={articleFilter === article}
                onClick={() => setArticleFilter((prev) => (prev === article ? null : article))}
              />
            ))}
          </div>
        </div>
      )}

      {/* Result count + clear */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {loading
            ? 'Loading…'
            : isSearching
              ? `${total} section${total !== 1 ? 's' : ''} match “${debouncedQuery}”`
              : hasFilters
                ? `${total} section${total !== 1 ? 's' : ''}`
                : `${total} section${total !== 1 ? 's' : ''} in the corpus`}
        </p>
        {hasFilters && (
          <Button variant="ghost" size="sm" className="text-xs" onClick={clearAll}>
            <X className="h-3 w-3 mr-1" /> Clear all filters
          </Button>
        )}
      </div>

      {/* Results */}
      {loading ? (
        <LoadingState message={isSearching ? 'Searching laws…' : 'Loading law library…'} />
      ) : error ? (
        <ErrorState message={error} onRetry={() => void load(0)} />
      ) : results.length === 0 ? (
        <EmptyState
          icon={<Scale className="h-6 w-6" />}
          title="No laws found"
          description="Try a different search term or clear the filters."
          action={
            hasFilters || isSearching ? (
              <Button variant="outline" size="sm" onClick={clearAll}>
                Clear search and filters
              </Button>
            ) : undefined
          }
        />
      ) : isSearching ? (
        /* Search mode — relevance-ordered flat list */
        <div className="space-y-4">
          {results.map((result) => (
            <LegalCitationCard key={result.law.id} law={result.law} />
          ))}
        </div>
      ) : (
        /* Browse mode — sections grouped by VTL article */
        <div className="space-y-8">
          {articleGroups.map(([article, laws]) => (
            <section key={article} className="space-y-3">
              <div className="flex items-center gap-3">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
                  <BookOpen className="h-4 w-4 text-primary" />
                  Article {article}
                </h2>
                <Badge variant="secondary" className="text-xs">
                  {laws.length} section{laws.length !== 1 ? 's' : ''}
                </Badge>
                <div className="h-px flex-1 bg-[#1e2d45]" />
              </div>
              <div className="space-y-4">
                {laws.map((law) => (
                  <LegalCitationCard key={law.id} law={law} />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}

      {/* Load more */}
      {!loading && !error && results.length < total && (
        <div className="flex justify-center">
          <Button variant="outline" onClick={() => void load(results.length)} disabled={loadingMore}>
            {loadingMore ? 'Loading…' : `Load more (${results.length} of ${total})`}
          </Button>
        </div>
      )}

      {/* Disclaimer */}
      {isMockMode ? (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-300">
          <strong>Demo mode:</strong> These law entries are sample demo text created solely for
          UI demonstration. Section identifiers are prefixed with “SAMPLE-” and do not represent
          real official law text. Do not use this library as legal reference.
        </div>
      ) : (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-300">
          <strong>Disclaimer:</strong> Provision text is loaded from the locally provisioned New
          York Vehicle &amp; Traffic Law corpus and is provided for informational purposes only.
          It is not legal advice; verify wording and current effect against the official NYS
          Vehicle &amp; Traffic Law before relying on it.
        </div>
      )}
    </div>
  );
}
