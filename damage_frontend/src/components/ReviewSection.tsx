import { Plus, Trash2 } from 'lucide-react'
import type { AnalysisRow, ConfigResponse } from '../types'

interface Props {
  rows: AnalysisRow[]
  config: ConfigResponse | null
  onRowsChange: (rows: AnalysisRow[]) => void
}

function updateRow(rows: AnalysisRow[], index: number, patch: Partial<AnalysisRow>): AnalysisRow[] {
  return rows.map((r, i) => (i === index ? { ...r, ...patch } : r))
}

export default function ReviewSection({ rows, config, onRowsChange }: Props) {
  const partChoices = config?.part_choices ?? []
  const damageTypes = config?.damage_types ?? []
  const severities = config?.severities ?? []

  function addRow() {
    const newRow: AnalysisRow = {
      part: partChoices[0] ?? '',
      damage_type: damageTypes[0] ?? '',
      severity: 'moderate',
      count: 1,
      confidence: null,
      needs_review: false,
      reviewed: true,
    }
    onRowsChange([...rows, newRow])
  }

  function deleteRow(index: number) {
    onRowsChange(rows.filter((_, i) => i !== index))
  }

  const selectCls =
    'border border-slate-300 rounded px-2 py-1 text-sm text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500 w-full'
  const inputCls =
    'border border-slate-300 rounded px-2 py-1 text-sm text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500 w-16'

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-800">Review Detected Damages</h2>
        <button
          onClick={addRow}
          className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-700 border border-blue-300 rounded-md px-3 py-1.5 hover:bg-blue-50 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Row
        </button>
      </div>

      {rows.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-lg p-8 text-center text-slate-400">
          No damage rows. Add one or go back and run the analysis.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-200 shadow-sm">
          <table className="min-w-full divide-y divide-slate-200 bg-white text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Part</th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Damage Type</th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Severity</th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Count</th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Confidence</th>
                <th className="px-3 py-2.5 text-center text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Flagged</th>
                <th className="px-3 py-2.5 text-center text-xs font-medium text-slate-600 uppercase tracking-wide whitespace-nowrap">Reviewed</th>
                <th className="px-3 py-2.5"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((row, i) => (
                <tr
                  key={i}
                  className={row.needs_review ? 'bg-yellow-50' : 'hover:bg-slate-50'}
                >
                  <td className="px-3 py-2">
                    {partChoices.length > 0 ? (
                      <select
                        value={row.part}
                        onChange={(e) => onRowsChange(updateRow(rows, i, { part: e.target.value }))}
                        className={selectCls}
                      >
                        {partChoices.map((p) => <option key={p} value={p}>{p}</option>)}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={row.part}
                        onChange={(e) => onRowsChange(updateRow(rows, i, { part: e.target.value }))}
                        className="border border-slate-300 rounded px-2 py-1 text-sm w-32"
                      />
                    )}
                  </td>
                  <td className="px-3 py-2">
                    {damageTypes.length > 0 ? (
                      <select
                        value={row.damage_type}
                        onChange={(e) => onRowsChange(updateRow(rows, i, { damage_type: e.target.value }))}
                        className={selectCls}
                      >
                        {damageTypes.map((d) => <option key={d} value={d}>{d}</option>)}
                      </select>
                    ) : (
                      <span className="text-slate-700">{row.damage_type}</span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    {severities.length > 0 ? (
                      <select
                        value={row.severity}
                        onChange={(e) => onRowsChange(updateRow(rows, i, { severity: e.target.value }))}
                        className={selectCls}
                      >
                        {severities.map((s) => <option key={s} value={s}>{s}</option>)}
                      </select>
                    ) : (
                      <span className="text-slate-700">{row.severity}</span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      min={1}
                      value={row.count}
                      onChange={(e) =>
                        onRowsChange(updateRow(rows, i, { count: Math.max(1, Number(e.target.value)) }))
                      }
                      className={inputCls}
                    />
                  </td>
                  <td className="px-3 py-2 text-slate-500">
                    {row.confidence !== null ? row.confidence.toFixed(2) : 'manual'}
                  </td>
                  <td className="px-3 py-2 text-center">
                    {/* Flagged: read-only */}
                    <input
                      type="checkbox"
                      checked={row.needs_review}
                      readOnly
                      disabled
                      className="cursor-not-allowed opacity-70"
                      aria-label="Flagged — read-only"
                    />
                  </td>
                  <td className="px-3 py-2 text-center">
                    {/* Reviewed: editable — but NOT auto-checked when needs_review=true */}
                    <input
                      type="checkbox"
                      checked={row.reviewed}
                      onChange={(e) =>
                        onRowsChange(updateRow(rows, i, { reviewed: e.target.checked }))
                      }
                      className="cursor-pointer"
                      aria-label="Reviewed"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <button
                      onClick={() => deleteRow(i)}
                      className="text-slate-400 hover:text-red-500 transition-colors"
                      aria-label="Delete row"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs text-slate-400">
        Rows highlighted in yellow are flagged and require manual review before calculating costs.
        Confidence and flagged status are read-only.
      </p>
    </div>
  )
}
