type Step = 'upload' | 'photos' | 'review' | 'cost'

interface Tab {
  id: Step
  label: string
}

interface Props {
  step: Step
  canViewResults: boolean
  onStepChange: (step: Step) => void
}

const TABS: Tab[] = [
  { id: 'upload', label: '1. Upload' },
  { id: 'photos', label: '2. Photos' },
  { id: 'review', label: '3. Review' },
  { id: 'cost', label: '4. Cost' },
]

export default function Navigation({ step, canViewResults, onStepChange }: Props) {
  return (
    <nav className="border-b border-slate-200 bg-white mb-6">
      <div className="flex gap-0">
        {TABS.map((tab) => {
          const isActive = step === tab.id
          const isDisabled = tab.id !== 'upload' && !canViewResults

          return (
            <button
              key={tab.id}
              disabled={isDisabled}
              onClick={() => !isDisabled && onStepChange(tab.id)}
              className={[
                'px-5 py-3 text-sm font-medium border-b-2 transition-colors',
                isActive
                  ? 'border-blue-600 text-slate-800'
                  : isDisabled
                    ? 'border-transparent text-slate-400 cursor-not-allowed'
                    : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300 cursor-pointer',
              ].join(' ')}
            >
              {tab.label}
            </button>
          )
        })}
      </div>
    </nav>
  )
}
