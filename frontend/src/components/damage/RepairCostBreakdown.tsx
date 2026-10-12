import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { DamageAssessmentRecord } from '@/types/damage';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { DemoBadge } from '@/components/common/DemoBadge';

interface RepairCostBreakdownProps {
  record: DamageAssessmentRecord;
}

const CATEGORY_COLORS: Record<string, string> = {
  parts: '#3b82f6',
  labor: '#8b5cf6',
  paint: '#f59e0b',
  other: '#64748b',
  tax: '#6b7280',
};

const CATEGORY_LABELS: Record<string, string> = {
  parts: 'Parts',
  labor: 'Labor',
  paint: 'Paint',
  other: 'Other',
  tax: 'Tax',
};

function formatCurrency(amount: number, currency: string = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(amount);
}

export function RepairCostBreakdown({ record }: RepairCostBreakdownProps) {
  const { repairItems, estimate } = record;

  // Derive per-category totals for chart
  const categoryTotals = repairItems.reduce<Record<string, number>>((acc, item) => {
    acc[item.category] = (acc[item.category] ?? 0) + item.total;
    return acc;
  }, {});

  const chartData = Object.entries(categoryTotals).map(([cat, total]) => ({
    category: CATEGORY_LABELS[cat] ?? cat,
    total,
    key: cat,
  }));

  return (
    <div className="space-y-6">
      {/* Chart */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Cost by Category</CardTitle>
            <DemoBadge />
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 4, right: 8, left: 8, bottom: 4 }}>
                <XAxis
                  dataKey="category"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => `$${v}`}
                />
                <Tooltip
                  contentStyle={{
                    background: '#131d35',
                    border: '1px solid #1e2d45',
                    borderRadius: 8,
                    color: '#e2e8f0',
                    fontSize: 12,
                  }}
                  formatter={(value: unknown) => [`$${(value as number).toLocaleString()}`, 'Total']}
                />
                <Bar dataKey="total" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.key}
                      fill={CATEGORY_COLORS[entry.key] ?? '#64748b'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-3 mt-3">
            {chartData.map((entry) => (
              <div key={entry.key} className="flex items-center gap-1.5 text-xs text-slate-400">
                <span
                  className="h-2.5 w-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: CATEGORY_COLORS[entry.key] ?? '#64748b' }}
                />
                {entry.category}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Line items table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Itemized Breakdown</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#1e2d45]">
                  <th className="text-left py-2.5 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="text-left py-2.5 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden sm:table-cell">
                    Category
                  </th>
                  <th className="text-right py-2.5 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Total
                  </th>
                </tr>
              </thead>
              <tbody>
                {repairItems.map((item) => (
                  <tr
                    key={item.id}
                    className="border-b border-[#1e2d45] last:border-0 hover:bg-[#131d35] transition-colors"
                  >
                    <td className="py-3 px-4 text-slate-200">{item.description}</td>
                    <td className="py-3 px-4 hidden sm:table-cell">
                      <Badge variant="secondary" className="text-xs capitalize">
                        {CATEGORY_LABELS[item.category] ?? item.category}
                      </Badge>
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-100">
                      {formatCurrency(item.total)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Summary */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Estimate Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Subtotal</span>
            <span className="text-slate-200 font-mono">{formatCurrency(estimate.subtotal)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Tax</span>
            <span className="text-slate-200 font-mono">{formatCurrency(estimate.tax)}</span>
          </div>
          <div className="flex justify-between text-sm font-semibold border-t border-[#1e2d45] pt-2 mt-2">
            <span className="text-slate-100">Total Estimate</span>
            <span className="text-primary font-mono text-base">
              {formatCurrency(estimate.total)}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Estimate Range</span>
            <span className="text-slate-300 font-mono">
              {formatCurrency(estimate.lowEstimate)} – {formatCurrency(estimate.highEstimate)}
            </span>
          </div>
          {estimate.egpEquivalent !== undefined && (
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">EGP Equivalent</span>
              <span className="text-slate-300 font-mono">
                EGP {estimate.egpEquivalent.toLocaleString()}
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Assumptions & disclaimer */}
      {estimate.assumptions.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Pricing Assumptions</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1">
              {estimate.assumptions.map((a, i) => (
                <li key={i} className="text-xs text-slate-400 flex gap-2">
                  <span className="shrink-0 text-primary">•</span>
                  {a}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-300">
        <strong>Disclaimer:</strong> This estimate is AI-generated from visual analysis only. Hidden
        damage may only be discovered during disassembly. Obtain quotes from at least three certified
        repair shops before authorizing any work.
      </div>
    </div>
  );
}
