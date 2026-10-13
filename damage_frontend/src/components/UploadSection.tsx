import { Loader2, X } from 'lucide-react'
import { useRef, useState } from 'react'

interface Props {
  onAnalyze: (formData: FormData) => void
  loading: boolean
  error: string | null
}

const ACCEPTED = ['.jpg', '.jpeg', '.png', '.webp']
const ACCEPTED_MIME = ['image/jpeg', 'image/png', 'image/webp']

function filterFiles(raw: FileList | File[]): File[] {
  return Array.from(raw).filter((f) =>
    ACCEPTED_MIME.includes(f.type) ||
    ACCEPTED.some((ext) => f.name.toLowerCase().endsWith(ext))
  )
}

export default function UploadSection({ onAnalyze, loading, error }: Props) {
  const [make, setMake] = useState('')
  const [modelName, setModelName] = useState('')
  const [year, setYear] = useState(2020)
  const [tier, setTier] = useState<'economy' | 'mid' | 'luxury'>('mid')
  const [useVlm, setUseVlm] = useState(true)
  const [useWeb, setUseWeb] = useState(false)
  const [files, setFiles] = useState<File[]>([])

  const fileInputRef = useRef<HTMLInputElement>(null)

  function addFiles(incoming: FileList | File[]) {
    const valid = filterFiles(incoming)
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name + f.size))
      const deduped = valid.filter((f) => !existing.has(f.name + f.size))
      return [...prev, ...deduped]
    })
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files)
  }

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files)
      e.target.value = ''
    }
  }

  function handleSubmit() {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f))
    fd.append('make', make.trim())
    fd.append('model_name', modelName.trim())
    fd.append('year', String(year))
    fd.append('tier', tier)
    fd.append('use_vlm', String(useVlm))
    fd.append('use_web', String(useWeb))
    onAnalyze(fd)
  }

  const canSubmit = files.length > 0 && make.trim() !== '' && modelName.trim() !== '' && !loading

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-slate-800">Car Damage Analysis</h1>
        <p className="text-slate-500 text-sm mt-1">
          Upload photos of a damaged vehicle to get an AI-powered damage assessment and repair cost estimate.
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form fields */}
        <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-6 space-y-4">
          <h2 className="font-medium text-slate-800 text-base">Vehicle Information</h2>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Make *</label>
              <input
                type="text"
                value={make}
                onChange={(e) => setMake(e.target.value)}
                placeholder="e.g. Toyota"
                className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Model *</label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g. Corolla"
                className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Year</label>
              <input
                type="number"
                value={year}
                min={1980}
                max={2026}
                onChange={(e) => setYear(Number(e.target.value))}
                className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Price Tier</label>
              <select
                value={tier}
                onChange={(e) => setTier(e.target.value as 'economy' | 'mid' | 'luxury')}
                className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="economy">Economy</option>
                <option value="mid">Mid</option>
                <option value="luxury">Luxury</option>
              </select>
            </div>
          </div>

          <div className="space-y-2 pt-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useVlm}
                onChange={(e) => setUseVlm(e.target.checked)}
                className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-slate-700">Use VLM (AI vision analysis)</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useWeb}
                onChange={(e) => setUseWeb(e.target.checked)}
                className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-slate-700">Look up replacement part prices online</span>
            </label>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={[
              'w-full mt-2 py-2.5 px-4 rounded-md text-sm font-medium flex items-center justify-center gap-2 transition-colors',
              canSubmit
                ? 'bg-blue-600 text-white hover:bg-blue-700'
                : 'bg-slate-200 text-slate-400 cursor-not-allowed',
            ].join(' ')}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Detecting damage and car parts...
              </>
            ) : (
              'Analyze Photos'
            )}
          </button>
        </div>

        {/* Dropzone */}
        <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-6 space-y-4">
          <h2 className="font-medium text-slate-800 text-base">Upload Photos</h2>

          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors"
          >
            <p className="text-slate-500 text-sm">
              Drag and drop images here, or click to browse
            </p>
            <p className="text-slate-400 text-xs mt-1">JPG, JPEG, PNG, WEBP — max 20 MB each</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              multiple
              className="hidden"
              onChange={handleFileInput}
            />
          </div>

          {files.length > 0 && (
            <div className="grid grid-cols-4 gap-2">
              {files.map((file, i) => (
                <div key={i} className="relative group">
                  <img
                    src={URL.createObjectURL(file)}
                    alt={file.name}
                    className="w-full h-20 object-cover rounded border border-slate-200"
                  />
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      removeFile(i)
                    }}
                    className="absolute top-0.5 right-0.5 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="w-3 h-3" />
                  </button>
                  <p className="text-xs text-slate-500 truncate mt-0.5 px-0.5">{file.name}</p>
                </div>
              ))}
            </div>
          )}

          {files.length === 0 && (
            <p className="text-xs text-slate-400 text-center">No photos selected</p>
          )}
        </div>
      </div>
    </div>
  )
}
