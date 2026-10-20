// Regression tests for null-safe numeric display formatting.
//
// The report page crashed with "Cannot read properties of null (reading
// 'toLocaleString')" because the API returns JSON null for optional fields
// (e.g. vehicleInfo.mileage) and the page called .toLocaleString() on them.
// These tests pin the formatter behaviour: unknown values must render as a
// fallback text and must never be misrepresented as zero.

import { describe, expect, it } from 'vitest';
import { formatNumber } from './format';

describe('formatNumber', () => {
  it('formats valid numbers with locale separators', () => {
    expect(formatNumber(36457.2)).toBe((36457.2).toLocaleString());
    expect(formatNumber(1250)).toBe((1250).toLocaleString());
  });

  it('formats zero as zero (a known value, not missing)', () => {
    expect(formatNumber(0)).toBe('0');
  });

  it('renders null, undefined and NaN as the fallback, never as 0', () => {
    expect(formatNumber(null)).toBe('N/A');
    expect(formatNumber(undefined)).toBe('N/A');
    expect(formatNumber(Number.NaN)).toBe('N/A');
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe('N/A');
  });

  it('supports a custom fallback text', () => {
    expect(formatNumber(null, 'Not available')).toBe('Not available');
  });

  it('mirrors the report-page guards: null mileage renders N/A (regression)', () => {
    // Shape copied from the live API: GET /api/damage/<id> with no mileage
    const record = {
      vehicleInfo: { make: 'Toyota', model: 'Camry', year: 2019, color: '', mileage: null },
    };
    const display =
      record.vehicleInfo.mileage != null
        ? `${formatNumber(record.vehicleInfo.mileage)} km`
        : 'N/A';
    expect(display).toBe('N/A');
  });

  it('renders a real mileage with the km suffix (regression)', () => {
    const record = {
      vehicleInfo: { make: 'Toyota', model: 'Camry', year: 2019, color: '', mileage: 62000 },
    };
    const display =
      record.vehicleInfo.mileage != null
        ? `${formatNumber(record.vehicleInfo.mileage)} km`
        : 'N/A';
    expect(display).toBe(`${(62000).toLocaleString()} km`);
  });
});
