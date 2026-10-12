import { useState, useEffect, useRef, type ChangeEvent } from 'react';
import { Check, Download, Printer, RotateCcw, LayoutDashboard } from 'lucide-react';
import { Link } from 'react-router-dom';
import { PageHeader } from '@/components/common/PageHeader';
import { DemoBadge } from '@/components/common/DemoBadge';
import { LoadingState } from '@/components/common/LoadingState';
import { ErrorState } from '@/components/common/ErrorState';
import { ImageUploader } from '@/components/upload/ImageUploader';
import { DamageAssessment } from '@/components/damage/DamageAssessment';
import { RepairCostBreakdown } from '@/components/damage/RepairCostBreakdown';
import { uploadDamageImages } from '@/services/damageService';
import { exportElementToPDF } from '@/utils/pdfExport';
import type { DamageAssessmentRecord, VehicleInfo } from '@/types/damage';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// ── Step metadata ──────────────────────────────────────────────────────────

const STEPS = [
  { id: 0, label: 'Upload Images' },
  { id: 1, label: 'Vehicle Info' },
  { id: 2, label: 'Damage Assessment' },
  { id: 3, label: 'Repair Cost' },
  { id: 4, label: 'Report' },
] as const;

type StepIndex = 0 | 1 | 2 | 3 | 4;

// ── Step Indicator ─────────────────────────────────────────────────────────

interface StepIndicatorProps {
  current: StepIndex;
}

function StepIndicator({ current }: StepIndicatorProps) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto pb-1">
      {STEPS.map((step, idx) => {
        const done = step.id < current;
        const active = step.id === current;

        return (
          <div key={step.id} className="flex items-center">
            <div
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors
                ${active ? 'bg-primary text-white' : ''}
                ${done ? 'text-green-400' : ''}
                ${!done && !active ? 'text-slate-500' : ''}
              `}
            >
              <span
                className={`h-5 w-5 rounded-full flex items-center justify-center text-xs shrink-0
                  ${active ? 'bg-white/20' : done ? 'bg-green-500/20' : 'bg-[#1e2d45]'}
                `}
              >
                {done ? <Check className="h-3 w-3" /> : step.id + 1}
              </span>
              <span className="hidden sm:inline">{step.label}</span>
            </div>
            {idx < STEPS.length - 1 && (
              <div
                className={`h-px w-4 ${step.id < current ? 'bg-green-500/40' : 'bg-[#1e2d45]'}`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Validation helpers ─────────────────────────────────────────────────────

type VehicleFormErrors = Partial<Record<keyof VehicleInfo, string>>;

function validateVehicleInfo(info: Partial<VehicleInfo>): VehicleFormErrors {
  const errs: VehicleFormErrors = {};
  if (!info.make?.trim()) errs.make = 'Make is required.';
  if (!info.model?.trim()) errs.model = 'Model is required.';
  if (!info.year) errs.year = 'Year is required.';
  else if (info.year < 1980 || info.year > 2026) errs.year = 'Year must be between 1980 and 2026.';
  if (!info.color?.trim()) errs.color = 'Color is required.';
  return errs;
}

// ── Main Page ──────────────────────────────────────────────────────────────

export function DamageEstimatorPage() {
  const reportRef = useRef<HTMLDivElement>(null);

  const [step, setStep] = useState<StepIndex>(0);
  const [imageFiles, setImageFiles] = useState<File[]>([]);

  const [vehicleInfo, setVehicleInfo] = useState<Partial<VehicleInfo>>({
    make: '',
    model: '',
    color: '',
    year: undefined,
    mileage: undefined,
    repairLocation: '',
  });
  const [vehicleErrors, setVehicleErrors] = useState<VehicleFormErrors>({});

  const [record, setRecord] = useState<DamageAssessmentRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // When step transitions to 2, trigger the assessment API call
  useEffect(() => {
    if (step !== 2) return;
    if (record) return; // already loaded

    let cancelled = false;

    async function fetchAssessment() {
      setLoading(true);
      setError(null);
      try {
        const result = await uploadDamageImages(
          imageFiles,
          vehicleInfo as VehicleInfo,
        );
        if (!cancelled) setRecord(result);
      } catch (err) {
        if (!cancelled)
          setError(err instanceof Error ? err.message : 'Assessment failed. Please try again.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void fetchAssessment();
    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  function resetAll() {
    setStep(0);
    setImageFiles([]);
    setVehicleInfo({ make: '', model: '', color: '', year: undefined, mileage: undefined, repairLocation: '' });
    setVehicleErrors({});
    setRecord(null);
    setLoading(false);
    setError(null);
  }

  function goNext() {
    if (step === 1) {
      const errs = validateVehicleInfo(vehicleInfo);
      if (Object.keys(errs).length > 0) {
        setVehicleErrors(errs);
        return;
      }
      setVehicleErrors({});
    }
    setStep((prev) => Math.min(prev + 1, 4) as StepIndex);
  }

  function goPrev() {
    setStep((prev) => Math.max(prev - 1, 0) as StepIndex);
  }

  function handleVehicleChange(field: keyof VehicleInfo, value: string | number | undefined) {
    setVehicleInfo((prev) => ({ ...prev, [field]: value }));
    if (vehicleErrors[field]) {
      setVehicleErrors((prev) => { const next = { ...prev }; delete next[field]; return next; });
    }
  }

  async function handleDownloadPDF() {
    if (!reportRef.current) {
      alert('PDF export: no report element found.');
      return;
    }
    await exportElementToPDF(reportRef.current, {
      filename: `damage-report-${record?.id ?? 'draft'}.pdf`,
      onError: (msg) => alert(msg),
    });
  }

  // ── Step renders ────────────────────────────────────────────────────────

  function renderStep() {
    switch (step) {
      case 0:
        return (
          <div className="space-y-6 max-w-2xl">
            <ImageUploader files={imageFiles} onFilesChange={setImageFiles} />
            <div className="flex justify-end">
              <Button onClick={goNext} disabled={imageFiles.length === 0} size="lg">
                Continue to Vehicle Info
              </Button>
            </div>
          </div>
        );

      case 1:
        return (
          <div className="max-w-xl space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Vehicle Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Make */}
                <div className="space-y-1.5">
                  <Label htmlFor="make">
                    Make <span className="text-red-400">*</span>
                  </Label>
                  <Input
                    id="make"
                    placeholder="e.g. Toyota"
                    value={vehicleInfo.make ?? ''}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      handleVehicleChange('make', e.target.value)
                    }
                    aria-invalid={!!vehicleErrors.make}
                  />
                  {vehicleErrors.make && (
                    <p className="text-xs text-red-400">{vehicleErrors.make}</p>
                  )}
                </div>

                {/* Model */}
                <div className="space-y-1.5">
                  <Label htmlFor="model">
                    Model <span className="text-red-400">*</span>
                  </Label>
                  <Input
                    id="model"
                    placeholder="e.g. Camry"
                    value={vehicleInfo.model ?? ''}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      handleVehicleChange('model', e.target.value)
                    }
                    aria-invalid={!!vehicleErrors.model}
                  />
                  {vehicleErrors.model && (
                    <p className="text-xs text-red-400">{vehicleErrors.model}</p>
                  )}
                </div>

                {/* Year */}
                <div className="space-y-1.5">
                  <Label htmlFor="year">
                    Year <span className="text-red-400">*</span>
                  </Label>
                  <Input
                    id="year"
                    type="number"
                    min={1980}
                    max={2026}
                    placeholder="e.g. 2020"
                    value={vehicleInfo.year ?? ''}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      handleVehicleChange('year', e.target.value ? parseInt(e.target.value, 10) : undefined)
                    }
                    aria-invalid={!!vehicleErrors.year}
                  />
                  {vehicleErrors.year && (
                    <p className="text-xs text-red-400">{vehicleErrors.year}</p>
                  )}
                </div>

                {/* Color */}
                <div className="space-y-1.5">
                  <Label htmlFor="color">
                    Color <span className="text-red-400">*</span>
                  </Label>
                  <Input
                    id="color"
                    placeholder="e.g. Silver"
                    value={vehicleInfo.color ?? ''}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      handleVehicleChange('color', e.target.value)
                    }
                    aria-invalid={!!vehicleErrors.color}
                  />
                  {vehicleErrors.color && (
                    <p className="text-xs text-red-400">{vehicleErrors.color}</p>
                  )}
                </div>

                {/* Mileage (optional) */}
                <div className="space-y-1.5">
                  <Label htmlFor="mileage">Mileage (optional, km)</Label>
                  <Input
                    id="mileage"
                    type="number"
                    min={0}
                    placeholder="e.g. 60000"
                    value={vehicleInfo.mileage ?? ''}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      handleVehicleChange(
                        'mileage',
                        e.target.value ? parseInt(e.target.value, 10) : undefined,
                      )
                    }
                  />
                </div>

                {/* Repair location (optional) */}
                <div className="space-y-1.5">
                  <Label htmlFor="repairLocation">Repair Location (optional)</Label>
                  <Input
                    id="repairLocation"
                    placeholder="e.g. Cairo"
                    value={vehicleInfo.repairLocation ?? ''}
                    onChange={(e: ChangeEvent<HTMLInputElement>) =>
                      handleVehicleChange('repairLocation', e.target.value)
                    }
                  />
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-between">
              <Button variant="outline" onClick={goPrev}>
                Back
              </Button>
              <Button onClick={goNext}>Continue to Damage Assessment</Button>
            </div>
          </div>
        );

      case 2:
        if (loading) return <LoadingState message="Analyzing damage from images…" />;
        if (error || !record)
          return (
            <ErrorState
              message={error ?? 'Assessment failed.'}
              onRetry={() => {
                setRecord(null);
                setError(null);
              }}
            />
          );
        return (
          <div className="space-y-6 max-w-3xl">
            <DamageAssessment record={record} />
            <div className="flex justify-between">
              <Button variant="outline" onClick={goPrev}>Back</Button>
              <Button onClick={goNext}>Continue to Repair Cost</Button>
            </div>
          </div>
        );

      case 3:
        if (!record) return <ErrorState message="No assessment data." onRetry={() => setStep(0)} />;
        return (
          <div className="space-y-6 max-w-3xl">
            <RepairCostBreakdown record={record} />
            <div className="flex justify-between">
              <Button variant="outline" onClick={goPrev}>Back</Button>
              <Button onClick={goNext}>Continue to Report</Button>
            </div>
          </div>
        );

      case 4:
        if (!record) return <ErrorState message="No assessment data." onRetry={() => setStep(0)} />;
        return (
          <div className="space-y-6 max-w-3xl">
            {/* Action buttons */}
            <div className="flex flex-wrap gap-2">
              <Button onClick={handleDownloadPDF} variant="default" size="sm">
                <Download className="h-4 w-4" /> Download PDF Report
              </Button>
              <Button onClick={() => window.print()} variant="outline" size="sm">
                <Printer className="h-4 w-4" /> Print
              </Button>
              <Button onClick={resetAll} variant="ghost" size="sm">
                <RotateCcw className="h-4 w-4" /> Start New Assessment
              </Button>
              <Button asChild variant="secondary" size="sm">
                <Link to="/dashboard">
                  <LayoutDashboard className="h-4 w-4" /> Back to Dashboard
                </Link>
              </Button>
            </div>

            {/* Report body */}
            <div ref={reportRef} className="space-y-6">
              <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
                <DemoBadge />
                <span>This report is generated from AI-analysis of uploaded images. Demo mode active.</span>
              </div>

              {/* Vehicle summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Vehicle Information</CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-2 text-sm">
                    <div>
                      <dt className="text-slate-400">Make / Model</dt>
                      <dd className="text-slate-100 font-medium">
                        {record.vehicleInfo.make} {record.vehicleInfo.model}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-slate-400">Year</dt>
                      <dd className="text-slate-100">{record.vehicleInfo.year}</dd>
                    </div>
                    <div>
                      <dt className="text-slate-400">Color</dt>
                      <dd className="text-slate-100">{record.vehicleInfo.color}</dd>
                    </div>
                    {record.vehicleInfo.mileage !== undefined && (
                      <div>
                        <dt className="text-slate-400">Mileage</dt>
                        <dd className="text-slate-100">{record.vehicleInfo.mileage.toLocaleString()} km</dd>
                      </div>
                    )}
                    {record.vehicleInfo.repairLocation && (
                      <div>
                        <dt className="text-slate-400">Repair Location</dt>
                        <dd className="text-slate-100">{record.vehicleInfo.repairLocation}</dd>
                      </div>
                    )}
                    <div>
                      <dt className="text-slate-400">Assessment Date</dt>
                      <dd className="text-slate-100">
                        {new Date(record.assessmentDate).toLocaleDateString()}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>

              {/* Image thumbnails */}
              {imageFiles.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Uploaded Images</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                      {imageFiles.map((file, i) => (
                        <img
                          key={i}
                          src={URL.createObjectURL(file)}
                          alt={`Vehicle image ${i + 1}`}
                          className="w-full aspect-square object-cover rounded-lg border border-[#1e2d45]"
                        />
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Damage summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Damage Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {record.damagedParts.map((part) => (
                      <div
                        key={part.id}
                        className="flex items-center justify-between text-sm py-1.5 border-b border-[#1e2d45] last:border-0"
                      >
                        <span className="text-slate-200">{part.name}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-slate-400 text-xs">{part.damageType}</span>
                          <Badge
                            variant={
                              part.severity === 'severe'
                                ? 'danger'
                                : part.severity === 'moderate'
                                ? 'warning'
                                : 'secondary'
                            }
                            className="text-xs capitalize"
                          >
                            {part.severity}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Cost summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Repair Cost Estimate</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid sm:grid-cols-2 gap-3">
                    <div className="rounded-lg bg-[#0d1525] border border-[#1e2d45] p-4 text-center">
                      <p className="text-xs text-slate-400 mb-1">Total Estimate</p>
                      <p className="text-2xl font-bold text-primary">
                        ${record.estimate.total.toLocaleString()}
                      </p>
                    </div>
                    <div className="rounded-lg bg-[#0d1525] border border-[#1e2d45] p-4 text-center">
                      <p className="text-xs text-slate-400 mb-1">Estimate Range</p>
                      <p className="text-lg font-semibold text-slate-200">
                        ${record.estimate.lowEstimate.toLocaleString()} –{' '}
                        ${record.estimate.highEstimate.toLocaleString()}
                      </p>
                    </div>
                  </div>
                  {record.estimate.egpEquivalent !== undefined && (
                    <p className="text-xs text-slate-500 text-center">
                      ≈ EGP {record.estimate.egpEquivalent.toLocaleString()} (demo rate)
                    </p>
                  )}
                </CardContent>
              </Card>

              {/* Limitations */}
              {record.estimate.limitations.length > 0 && (
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3">
                  <p className="text-xs font-semibold text-amber-400 mb-2">Limitations</p>
                  <ul className="space-y-1">
                    {record.estimate.limitations.map((l, i) => (
                      <li key={i} className="text-xs text-amber-300 flex gap-2">
                        <span className="shrink-0">•</span>
                        {l}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        );
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Damage & Repair Estimator"
        description="Upload vehicle images to assess damage and estimate repair costs"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Damage Estimator' },
        ]}
        actions={<DemoBadge />}
      />

      <StepIndicator current={step} />

      <div className="pt-2">{renderStep()}</div>
    </div>
  );
}
