// -----------------------------------------------------------------------------
// NEER Frontend — shared display formatters  (Phase 34E)
//
// Small, presentation-only helpers used by more than one dashboard component
// (components/inspection/PointInspection.js and
// dashboard/_components/DashboardKPIs.js both need to format the same
// "selected date" and "signed anomaly" values). Pulled out of
// PointInspection.js — where they originated in Phase 34D — so the KPI
// drawer added in Phase 34E shares one implementation instead of carrying a
// second copy that could quietly drift out of sync with the first.
//
// Framework-independent (no React), same convention as lib/oceanDomain.js.
// -----------------------------------------------------------------------------

const DATE_FORMAT = { day: "2-digit", month: "short", year: "numeric" };

/**
 * Formats a date for display, e.g. "18 Mar 2024".
 * @param {string|Date|null|undefined} date
 * @returns {string} "No date selected" for falsy input; the raw input
 *   coerced to a string if it can't be parsed as a date.
 */
export function formatDate(date) {
  if (!date) return "No date selected";
  const parsed = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(parsed.getTime())) return String(date);
  return parsed.toLocaleDateString("en-GB", DATE_FORMAT);
}

/**
 * Formats a number with an explicit "+" sign for positive values, e.g.
 * "+0.42" / "-1.10" — for anomalies, where the sign is scientifically
 * meaningful and shouldn't rely on the reader noticing its absence.
 * @param {number} value
 * @param {number} [digits=2]
 * @returns {string} "--" when `value` isn't a finite number.
 */
export function formatSigned(value, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}