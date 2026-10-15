import type { VideoBackend } from '@/types/pipeline';

const STORAGE_KEY = 'accident-intel-settings';
export interface Settings { videoBackend: VideoBackend }
export const DEFAULT_SETTINGS: Settings = { videoBackend: 'auto' };

export function getSettings(): Settings {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    const videoBackend = parsed?.videoBackend;
    return { videoBackend: ['auto', 'local', 'openai'].includes(videoBackend) ? videoBackend : 'auto' };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(updates: Partial<Settings>): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...getSettings(), ...updates }));
}

export function resetSettings(): void { localStorage.removeItem(STORAGE_KEY); }
