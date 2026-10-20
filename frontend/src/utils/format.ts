// Shared display-formatting helpers

/**
 * Null-safe numeric formatting for display.
 *
 * Formats `value` with the locale's thousands separators. Unknown values
 * (null, undefined) and non-finite numbers (NaN, Infinity) render as
 * `fallback` so they are never misrepresented as zero.
 *
 * @param value - Number to format, or null/undefined when unknown.
 * @param fallback - Text shown when the value is unknown (default "N/A").
 * @returns Localized number string, or the fallback text.
 */
export function formatNumber(
  value: number | null | undefined,
  fallback = 'N/A',
): string {
  return value == null || !Number.isFinite(value)
    ? fallback
    : value.toLocaleString();
}
