// Settings service — reads and writes to localStorage
// No mock/real distinction needed: settings are always local.

const STORAGE_KEY = 'accident-intel-settings';

// ---------------------------------------------------------------------------
// Settings shape
// ---------------------------------------------------------------------------

export interface Settings {
  /** UI color theme */
  theme: 'dark' | 'light' | 'system';
  /** Primary jurisdiction for law retrieval */
  jurisdiction: string;
  /** Display currency */
  currency: 'USD' | 'EGP';
  /** Speed unit for vehicle tracking display */
  speedUnit: 'mph' | 'kph';
  /** Maximum upload file size in megabytes */
  maxUploadMB: number;
  /** Number of days to retain analysis data locally */
  retainDataDays: number;
  /** Whether to enable browser notification badges */
  notificationsEnabled: boolean;
  /** Auto-save intermediate analysis results */
  autoSave: boolean;
  /** Preferred report export format */
  reportFormat: 'pdf' | 'markdown';
}

const DEFAULT_SETTINGS: Settings = {
  theme: 'dark',
  jurisdiction: 'NYC',
  currency: 'USD',
  speedUnit: 'mph',
  maxUploadMB: 500,
  retainDataDays: 90,
  notificationsEnabled: true,
  autoSave: true,
  reportFormat: 'pdf',
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Read settings from localStorage, falling back to defaults for any missing key */
export function getSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

/** Persist a partial settings update; merges with existing settings */
export function saveSettings(updates: Partial<Settings>): void {
  const current = getSettings();
  const merged: Settings = { ...current, ...updates };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
}

/** Reset all settings to factory defaults */
export function resetSettings(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export { DEFAULT_SETTINGS };
