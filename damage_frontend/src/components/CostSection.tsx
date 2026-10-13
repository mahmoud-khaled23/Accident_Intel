import { AlertTriangle, Download, Loader2 } from 'lucide-react'
import type { AnalysisRow, CostResponse } from '../types'

interface Props {
  rows: AnalysisRow[]
  analysisId: string | null
  make: string
  modelName: string
  year: number
  tier: string
  costResult: CostResponse | null
  costStale: boolean
  loading: boolean
  error: string | null
  onCalculate: () => void
}

function fmtRange(lo: number, hi: number): string {
  if (lo === 0 && hi === 0) return '—'
  if (lo === hi) return lo.toLocaleString()
  return `${lo.toLocaleString()} – ${hi.toLocaleString()}`
}

function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function CostSection({
  costResult,
  costStale,
  loading,
  error,
  onCalculate,
}: Props) {
  const report = costResult?.cost ?? null

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-800">Cost Estimate</h2>
        <button
          onClick={onCalculate}
          disabled={loading}
          className={[
            'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
            loading
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700',
          ].join(' ')}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Calculating...
            </>
          ) : (
            'Calculate Cost'
          )}
        </button>
      </div>

      {/* Stale banner */}
      {costStale && (
        <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-3 text-yellow-700 text-sm flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          Damages changed since last estimate — recalculate to refresh results.
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      {report && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
              <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Total Cost (EGP)</p>
              <p className="text-2xl font-bold text-slate-800">
                {fmtRange(report.total_min, report.total_max)}
              </p>
              <p className="text-xs text-slate-400 mt-0.5">{report.currency}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
              <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Unconfirmed (EGP)</p>
              <p className="text-2xl font-bold text-yellow-600">
                {fmtRange(report.unconfirmed_min, report.unconfirmed_max)}
              </p>
              <p className="text-xs text-slate-400 mt-0.5">pending review</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
              <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Parts to Work On</p>
              <p className="text-2xl font-bold text-slate-800">{report.lines.length}</p>
              <p className="text-xs text-slate-400 mt-0.5">line items</p>
            </div>
          </div>

          {/* Itemized table */}
          <div className="overflow-x-auto rounded-lg border border-slate-200 shadow-sm">
            <table className="min-w-full divide-y divide-slate-200 bg-white text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">Part</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">Work</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">Damages</th>
                  <th className="px-3 py-2.5 text-right text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">New Part</th>
                  <th className="px-3 py-2.5 text-right text-xs font-medium text-slate-600 uppercase tracking-wide">Labor</th>
                  <th className="px-3 py-2.5 text-right text-xs font-medium text-slate-600 uppercase tracking-wide">Paint</th>
                  <th className="px-3 py-2.5 text-right text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Total (EGP)</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Price Source</th>
                  <th className="px-3 py-2.5 text-center text-xs font-medium text-slate-600 uppercase tracking-wide">Unconfirmed</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {report.lines.map((line, i) => (
                  <tr
                    key={i}
                    className={[
                      line.needs_review ? 'bg-yellow-50 italic' : 'hover:bg-slate-50',
                    ].join(' ')}
                  >
                    <td className="px-3 py-2 text-slate-800 font-medium">{line.part}</td>
                    <td className="px-3 py-2 text-slate-600">{line.action}</td>
                    <td className="px-3 py-2 text-slate-500 text-xs">{line.damages.join(', ')}</td>
                    <td className="px-3 py-2 text-right text-slate-700 whitespace-nowrap">
                      {fmtRange(line.parts_min, line.parts_max)}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-700 whitespace-nowrap">
                      {fmtRange(line.labor_min, line.labor_max)}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-700 whitespace-nowrap">
                      {fmtRange(line.paint_min, line.paint_max)}
                    </td>
                    <td className="px-3 py-2 text-right font-medium text-slate-800 whitespace-nowrap">
                      {fmtRange(line.total_min, line.total_max)}
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-500">{line.price_source}</td>
                    <td className="px-3 py-2 text-center">
                      {line.needs_review && (
                        <span className="bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded text-xs">
                          Yes
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Notes, unpriced, warnings */}
          <div className="space-y-3">
            {report.unpriced.length > 0 && (
              <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                <p className="text-sm font-medium text-orange-700 mb-1">Unpriced Parts</p>
                <ul className="text-sm text-orange-600 list-disc list-inside space-y-0.5">
                  {report.unpriced.map((u, i) => <li key={i}>{u}</li>)}
                </ul>
              </div>
            )}

            {costResult && costResult.warnings.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm font-medium text-yellow-700 mb-1">Warnings</p>
                <ul className="text-sm text-yellow-600 list-disc list-inside space-y-0.5">
                  {costResult.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}

            {report.notes.length > 0 && (
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <p className="text-sm font-medium text-slate-600 mb-1">Notes</p>
                <ul className="text-sm text-slate-500 list-disc list-inside space-y-0.5">
                  {report.notes.map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </div>
            )}
          </div>

          {/* Download button */}
          <div className="flex justify-end">
            <button
              onClick={() => downloadJson(report, 'cost.json')}
              className="flex items-center gap-2 text-sm text-slate-600 hover:text-slate-800 border border-slate-300 rounded-md px-4 py-2 hover:bg-slate-50 transition-colors"
            >
              <Download className="w-4 h-4" />
              Download JSON
            </button>
          </div>
        </>
      )}
    </div>
  )
}
