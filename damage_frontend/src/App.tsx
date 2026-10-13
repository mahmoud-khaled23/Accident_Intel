import { useState } from 'react'
import CostSection from './components/CostSection'
import Navigation from './components/Navigation'
import PhotoSection from './components/PhotoSection'
import ReviewSection from './components/ReviewSection'
import UploadSection from './components/UploadSection'
import { useAppState } from './hooks/useAppState'
import { analyzePhotos, estimateCost } from './services/api'

interface CarInfo {
  make: string
  modelName: string
  year: number
  tier: string
}

export default function App() {
  const {
    step,
    analysisResult,
    editedRows,
    costResult,
    costStale,
    loading,
    error,
    config,
    goToStep,
    setAnalysisResult,
    setEditedRows,
    setCostResult,
    setLoading,
    setError,
  } = useAppState()

  const [carInfo, setCarInfo] = useState<CarInfo>({
    make: '',
    modelName: '',
    year: 2020,
    tier: 'mid',
  })

  const canViewResults = analysisResult !== null

  async function handleAnalyze(formData: FormData) {
    setLoading(true)
    setError(null)

    // Extract car info from the formData to keep it in sync
    const make = (formData.get('make') as string) ?? ''
    const modelName = (formData.get('model_name') as string) ?? ''
    const year = Number(formData.get('year') ?? 2020)
    const tier = (formData.get('tier') as string) ?? 'mid'
    setCarInfo({ make, modelName, year, tier })

    try {
      const result = await analyzePhotos(formData)
      setAnalysisResult(result)
      goToStep('photos')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  async function handleCalculateCost() {
    if (!analysisResult) return
    setLoading(true)
    setError(null)

    try {
      const result = await estimateCost({
        analysis_id: analysisResult.analysis_id,
        rows: editedRows,
        make: carInfo.make,
        model_name: carInfo.modelName,
        year: carInfo.year,
        tier: carInfo.tier,
        use_web: false,
      })
      setCostResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 px-4 py-3">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-base font-semibold text-slate-800">
            Accident Intel — Damage Estimator
          </h1>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <Navigation
          step={step}
          canViewResults={canViewResults}
          onStepChange={goToStep}
        />

        {step === 'upload' && (
          <UploadSection
            onAnalyze={handleAnalyze}
            loading={loading}
            error={error}
          />
        )}

        {step === 'photos' && analysisResult && (
          <PhotoSection
            photos={analysisResult.photos}
            warnings={analysisResult.warnings}
          />
        )}

        {step === 'review' && (
          <ReviewSection
            rows={editedRows}
            config={config}
            onRowsChange={setEditedRows}
          />
        )}

        {step === 'cost' && (
          <CostSection
            rows={editedRows}
            analysisId={analysisResult?.analysis_id ?? null}
            make={carInfo.make}
            modelName={carInfo.modelName}
            year={carInfo.year}
            tier={carInfo.tier}
            costResult={costResult}
            costStale={costStale}
            loading={loading}
            error={error}
            onCalculate={handleCalculateCost}
          />
        )}
      </div>
    </div>
  )
}
