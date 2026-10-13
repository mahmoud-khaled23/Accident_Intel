import { useState, useRef } from 'react';
import { User, Palette, Cpu, FileText, Upload, ShieldAlert } from 'lucide-react';
import { PageHeader } from '@/components/common/PageHeader';
import { DemoBadge } from '@/components/common/DemoBadge';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import {
  getSettings,
  saveSettings,
  resetSettings,
  type Settings,
} from '@/services/settingsService';

// ── Saved indicator ────────────────────────────────────────────────────────

function useSavedIndicator() {
  const [saved, setSaved] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function trigger() {
    setSaved(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setSaved(false), 2000);
  }

  return { saved, trigger };
}

// ── Section wrapper ────────────────────────────────────────────────────────

interface SectionProps {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  onSave: () => void;
  saved: boolean;
}

function Section({ icon, title, children, onSave, saved }: SectionProps) {
  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <span className="text-primary">{icon}</span>
          {title}
        </CardTitle>
      </CardHeader>
      <Separator />
      <CardContent className="pt-4 space-y-4">
        {children}
        <div className="flex items-center gap-3 pt-2">
          <Button size="sm" onClick={onSave}>
            Save
          </Button>
          {saved && (
            <span className="text-xs text-green-400 transition-opacity">Saved ✓</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────

export function SettingsPage() {
  const [settings, setSettings] = useState<Settings>(() => getSettings());
  const [confirmResetOpen, setConfirmResetOpen] = useState(false);
  const [confirmClearOpen, setConfirmClearOpen] = useState(false);
  const [displayName, setDisplayName] = useState<string>(
    () => getSettings().displayName ?? 'Demo User',
  );

  // Per-section saved indicators
  const profileSaved = useSavedIndicator();
  const appearanceSaved = useSavedIndicator();
  const analysisSaved = useSavedIndicator();
  const reportSaved = useSavedIndicator();
  const uploadSaved = useSavedIndicator();
  const privacySaved = useSavedIndicator();

  function update<K extends keyof Settings>(key: K, value: Settings[K]) {
    setSettings((prev) => ({ ...prev, [key]: value }));
  }

  function saveSection(updates: Partial<Settings>, indicator: { trigger: () => void }) {
    saveSettings(updates);
    indicator.trigger();
  }

  function handleResetAll() {
    resetSettings();
    setSettings(getSettings());
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Manage application preferences and configuration"
        breadcrumbs={[
          { label: 'Dashboard', href: '/dashboard' },
          { label: 'Settings' },
        ]}
        actions={<DemoBadge />}
      />

      <div className="max-w-2xl space-y-6">
        {/* 1. Profile */}
        <Section
          icon={<User className="h-4 w-4" />}
          title="Profile"
          onSave={() => saveSection({ displayName }, profileSaved)}
          saved={profileSaved.saved}
        >
          <div className="flex items-center gap-4">
            <div className="h-14 w-14 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center text-xl font-bold text-primary select-none">
              U
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-200">{displayName || 'Demo User'}</p>
              <p className="text-xs text-slate-500">user@example.com</p>
            </div>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="displayName">Display Name</Label>
            <Input
              id="displayName"
              placeholder="Your display name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="email">Email (read-only)</Label>
            <Input id="email" value="user@example.com" readOnly className="opacity-60 cursor-not-allowed" />
          </div>
        </Section>

        {/* 2. Appearance */}
        <Section
          icon={<Palette className="h-4 w-4" />}
          title="Appearance"
          onSave={() => saveSection({ theme: settings.theme }, appearanceSaved)}
          saved={appearanceSaved.saved}
        >
          <div className="space-y-1.5">
            <Label htmlFor="theme">Theme</Label>
            <Select
              value={settings.theme}
              onValueChange={(v) => update('theme', v as Settings['theme'])}
            >
              <SelectTrigger id="theme" className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="system">System</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-slate-500">
              Note: theme switching is saved but not yet applied — a future enhancement.
            </p>
          </div>
        </Section>

        {/* 3. Analysis Defaults */}
        <Section
          icon={<Cpu className="h-4 w-4" />}
          title="Analysis Defaults"
          onSave={() =>
            saveSection({ jurisdiction: settings.jurisdiction, speedUnit: settings.speedUnit }, analysisSaved)
          }
          saved={analysisSaved.saved}
        >
          <div className="space-y-1.5">
            <Label htmlFor="jurisdiction">Default Jurisdiction</Label>
            <Input
              id="jurisdiction"
              value={settings.jurisdiction}
              onChange={(e) => update('jurisdiction', e.target.value)}
              placeholder="e.g. NYC"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="speedUnit">Speed Unit</Label>
            <Select
              value={settings.speedUnit}
              onValueChange={(v) => update('speedUnit', v as Settings['speedUnit'])}
            >
              <SelectTrigger id="speedUnit" className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mph">mph</SelectItem>
                <SelectItem value="kph">km/h</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </Section>

        {/* 4. Report Defaults */}
        <Section
          icon={<FileText className="h-4 w-4" />}
          title="Report Defaults"
          onSave={() =>
            saveSection({ reportFormat: settings.reportFormat, currency: settings.currency }, reportSaved)
          }
          saved={reportSaved.saved}
        >
          <div className="space-y-1.5">
            <Label htmlFor="reportFormat">Default Report Format</Label>
            <Select
              value={settings.reportFormat}
              onValueChange={(v) => update('reportFormat', v as Settings['reportFormat'])}
            >
              <SelectTrigger id="reportFormat" className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pdf">PDF</SelectItem>
                <SelectItem value="markdown">Markdown</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="currency">Default Currency</Label>
            <Select
              value={settings.currency}
              onValueChange={(v) => update('currency', v as Settings['currency'])}
            >
              <SelectTrigger id="currency" className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="USD">USD — US Dollar</SelectItem>
                <SelectItem value="EGP">EGP — Egyptian Pound</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </Section>

        {/* 5. Upload Preferences */}
        <Section
          icon={<Upload className="h-4 w-4" />}
          title="Upload Preferences"
          onSave={() =>
            saveSection({ maxUploadMB: settings.maxUploadMB, autoSave: settings.autoSave }, uploadSaved)
          }
          saved={uploadSaved.saved}
        >
          <div className="rounded-lg bg-[#0d1525] border border-[#1e2d45] px-4 py-3 text-xs text-slate-400 space-y-1">
            <p><span className="text-slate-300 font-medium">Accepted formats:</span> MP4, MOV, WebM (video) · JPG, PNG, WebP (images)</p>
            <p><span className="text-slate-300 font-medium">Max video size:</span> 500 MB · <span className="text-slate-300 font-medium">Max image size:</span> 20 MB</p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="maxUploadMB">Max Upload Size (MB)</Label>
            <Input
              id="maxUploadMB"
              type="number"
              min={50}
              max={2000}
              value={settings.maxUploadMB}
              onChange={(e) => update('maxUploadMB', parseInt(e.target.value, 10) || 500)}
              className="w-40"
            />
          </div>
          <div className="flex items-center gap-3">
            <Switch
              id="autoSave"
              checked={settings.autoSave}
              onCheckedChange={(v) => update('autoSave', v)}
            />
            <Label htmlFor="autoSave" className="cursor-pointer">
              Auto-save intermediate results
            </Label>
          </div>
        </Section>

        {/* 6. Privacy & Data */}
        <Section
          icon={<ShieldAlert className="h-4 w-4" />}
          title="Privacy & Data"
          onSave={() =>
            saveSection(
              {
                retainDataDays: settings.retainDataDays,
                notificationsEnabled: settings.notificationsEnabled,
              },
              privacySaved,
            )
          }
          saved={privacySaved.saved}
        >
          <div className="rounded-lg bg-[#0d1525] border border-[#1e2d45] px-4 py-3 text-xs text-slate-400">
            All analysis data is stored locally in your browser. No data is sent to external servers
            in demo mode.
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="retainDataDays">Data Retention (days)</Label>
            <Input
              id="retainDataDays"
              type="number"
              min={1}
              max={365}
              value={settings.retainDataDays}
              onChange={(e) => update('retainDataDays', parseInt(e.target.value, 10) || 90)}
              className="w-40"
            />
          </div>

          <div className="flex items-center gap-3">
            <Switch
              id="notifications"
              checked={settings.notificationsEnabled}
              onCheckedChange={(v) => update('notificationsEnabled', v)}
            />
            <Label htmlFor="notifications" className="cursor-pointer">
              Enable notification badges
            </Label>
          </div>

          <Separator />

          <div className="space-y-3">
            <Button
              variant="outline"
              size="sm"
              className="border-red-500/40 text-red-400 hover:bg-red-500/10"
              onClick={() => setConfirmClearOpen(true)}
            >
              Clear local analysis history
            </Button>

            <Button
              variant="ghost"
              size="sm"
              className="text-red-400 hover:bg-red-500/10"
              onClick={() => setConfirmResetOpen(true)}
            >
              Reset all settings to defaults
            </Button>
          </div>
        </Section>
      </div>

      {/* Confirm: clear local history */}
      <ConfirmDialog
        open={confirmClearOpen}
        onOpenChange={setConfirmClearOpen}
        title="Clear Analysis History"
        description="This will remove all locally stored analysis records. This action cannot be undone."
        confirmLabel="Clear History"
        variant="danger"
        onConfirm={() => {
          // Clear all localStorage keys used by the app
          const keysToRemove = Object.keys(localStorage).filter(
            (k) => k.startsWith('accident-intel'),
          );
          keysToRemove.forEach((k) => localStorage.removeItem(k));
          setConfirmClearOpen(false);
        }}
      />

      {/* Confirm: reset all settings */}
      <ConfirmDialog
        open={confirmResetOpen}
        onOpenChange={setConfirmResetOpen}
        title="Reset All Settings"
        description="All settings will be restored to their factory defaults. This cannot be undone."
        confirmLabel="Reset Settings"
        variant="danger"
        onConfirm={handleResetAll}
      />
    </div>
  );
}
