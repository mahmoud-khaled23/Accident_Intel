import { useState, useEffect, useCallback } from 'react';
import { DollarSign } from 'lucide-react';
import { PageHeader } from '@/components/common/PageHeader';
import { EmptyState } from '@/components/common/EmptyState';
import { ErrorState } from '@/components/common/ErrorState';
import { LoadingState } from '@/components/common/LoadingState';
import { Badge } from '@/components/ui/badge';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import { getCostAnalysis } from '@/services/costService';
import type { CostAnalysis, ModelSource } from '@/types/cost';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const usdFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 6,
});

function formatUsd(value: number): string {
  return usdFormatter.format(value);
}

function formatNumber(value: number): string {
  return value.toLocaleString('en-US');
}

function sourceBadgeVariant(source: ModelSource): 'default' | 'secondary' | 'outline' {
  if (source === 'api') return 'default';
  if (source === 'local') return 'secondary';
  return 'outline';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CostAnalysisPage() {
  const [data, setData] = useState<CostAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getCostAnalysis();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cost analysis.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Cost Analysis"
        description="Per-model token usage and USD cost for all pipeline model calls"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Cost Analysis' },
        ]}
      />

      {loading ? (
        <LoadingState message="Loading cost analysis…" />
      ) : error ? (
        <ErrorState message={error} onRetry={() => void load()} />
      ) : !data || data.models.length === 0 ? (
        <EmptyState
          icon={<DollarSign className="h-6 w-6" />}
          title="No model calls recorded yet"
          description="No model calls recorded yet. Run an analysis to start populating cost data."
        />
      ) : (
        <>
          {/* Grand total summary cards */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Total Calls
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-semibold text-slate-100">
                  {formatNumber(data.grand_total.calls)}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Total Tokens
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-semibold text-slate-100">
                  {formatNumber(data.grand_total.total_tokens)}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Total Cost (USD)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-semibold text-slate-100">
                  {formatUsd(data.grand_total.cost_usd)}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Per-model table */}
          <Card>
            <CardHeader>
              <CardTitle>Model Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead className="text-right">Calls</TableHead>
                    <TableHead className="text-right">Prompt tokens</TableHead>
                    <TableHead className="text-right">Completion tokens</TableHead>
                    <TableHead className="text-right">Total tokens</TableHead>
                    <TableHead className="text-right">Cost USD</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.models.map((row) => (
                    <TableRow key={`${row.model_id}-${row.source}`}>
                      <TableCell className="font-mono text-xs">{row.model_id}</TableCell>
                      <TableCell>
                        <Badge variant={sourceBadgeVariant(row.source)}>
                          {row.source}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">{formatNumber(row.calls)}</TableCell>
                      <TableCell className="text-right">{formatNumber(row.prompt_tokens)}</TableCell>
                      <TableCell className="text-right">{formatNumber(row.completion_tokens)}</TableCell>
                      <TableCell className="text-right">{formatNumber(row.total_tokens)}</TableCell>
                      <TableCell className="text-right font-mono">{formatUsd(row.cost_usd)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Amber disclaimer banner for the note */}
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-300">
            <strong>Note:</strong> {data.note}
          </div>
        </>
      )}
    </div>
  );
}
