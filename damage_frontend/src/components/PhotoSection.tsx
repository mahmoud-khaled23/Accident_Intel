import { AlertTriangle } from 'lucide-react'
import { useState } from 'react'
import type { PhotoResult } from '../types'

type ImageTab = 'original' | 'combined' | 'parts' | 'damage'

interface Props {
  photos: PhotoResult[]
  warnings: string[]
}

const IMAGE_TABS: { id: ImageTab; label: string }[] = [
  { id: 'original', label: 'Original' },
  { id: 'combined', label: 'Combined' },
  { id: 'parts', label: 'Parts' },
  { id: 'damage', label: 'Damage' },
]

function severityClass(severity: string): string {
  switch (severity.toLowerCase()) {
    case 'minor':
      return 'bg-green-100 text-green-700'
    case 'moderate':
      return 'bg-yellow-100 text-yellow-700'
    case 'severe':
      return 'bg-red-100 text-red-700'
    default:
      return 'bg-slate-100 text-slate-600'
  }
}

function PhotoCard({ photo }: { photo: PhotoResult }) {
  const [activeTab, setActiveTab] = useState<ImageTab>('original')

  const imgUrl = photo.image_urls[activeTab]

  if (!photo.usable) {
    return (
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
        <div className="flex items-center gap-2 text-orange-700 font-medium mb-2">
          <AlertTriangle className="w-4 h-4" />
          <span>{photo.photo_id} — Unusable photo</span>
        </div>
        {photo.notes.length > 0 && (
          <ul className="text-sm text-orange-600 list-disc list-inside">
            {photo.notes.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        )}
      </div>
    )
  }

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm overflow-hidden">
      <div className="px-4 pt-4 pb-2">
        <p className="text-sm font-medium text-slate-700">{photo.photo_id}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-0">
        {/* Image viewer */}
        <div className="border-r border-slate-100">
          <div className="flex border-b border-slate-100 bg-slate-50">
            {IMAGE_TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={[
                  'px-3 py-1.5 text-xs font-medium transition-colors',
                  activeTab === tab.id
                    ? 'bg-white border-b-2 border-blue-600 text-blue-600'
                    : 'text-slate-500 hover:text-slate-700',
                ].join(' ')}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="p-2 bg-slate-50 min-h-48 flex items-center justify-center">
            {imgUrl ? (
              <img
                src={imgUrl}
                alt={`${photo.photo_id} — ${activeTab}`}
                className="max-h-64 w-full object-contain rounded"
              />
            ) : (
              <p className="text-xs text-slate-400">Image not available</p>
            )}
          </div>
        </div>

        {/* Detection list */}
        <div className="p-4 overflow-y-auto max-h-80">
          {photo.items.length === 0 ? (
            <p className="text-sm text-slate-400 italic">No damage detected</p>
          ) : (
            <ul className="space-y-3">
              {photo.items.map((item, i) => (
                <li key={i} className="text-sm">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="font-medium text-slate-800">{item.part}</span>
                    <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded text-xs">
                      {item.damage_type}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-xs ${severityClass(item.severity)}`}>
                      {item.severity}
                    </span>
                    <span className="text-slate-400 text-xs">
                      {(item.confidence * 100).toFixed(0)}%
                    </span>
                    {item.part_source && (
                      <span className="bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded text-xs">
                        {item.part_source}
                      </span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}

          {photo.notes.length > 0 && (
            <ul className="mt-3 text-xs text-slate-500 list-disc list-inside space-y-1">
              {photo.notes.map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

export default function PhotoSection({ photos, warnings }: Props) {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-800">Photos &amp; Detection Results</h2>

      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((w, i) => (
            <div key={i} className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-yellow-700 text-sm flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      <div className="space-y-4">
        {photos.map((photo) => (
          <PhotoCard key={photo.photo_id} photo={photo} />
        ))}
      </div>
    </div>
  )
}
