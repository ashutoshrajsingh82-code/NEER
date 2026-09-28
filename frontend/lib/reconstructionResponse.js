// -----------------------------------------------------------------------------
// NEER Frontend — point-reconstruction response validation  (Phase 37A)
//
// Pure, framework-independent validation of a `GET /reconstruct` body
// (backend `PointReconstructionResponse`, backend/app/schemas.py). Kept out
// of api.js so it can be unit-tested without fetch, and so api.js can import
// it without a cycle (same arrangement as dateDepthModel.js).
//
// What counts as "complete" is what the existing consumers actually read —
// PointInspection (temperature, anomaly, data_mode, cache_hit, latency_ms,
// notes) — plus every other non-nullable field of the backend schema. The
// backend declares `temperature`/`anomaly` as non-null floats, so a null or
// non-finite value there is a broken response, never a "no data" signal, and
// is rejected rather than coerced or defaulted.
//
// Nothing here fabricates or repairs a value: a body either passes unchanged
// or is rejected with a reason.
// -----------------------------------------------------------------------------

import { isIsoDate } from "./dateDepthModel.js";

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);
const isNonEmptyString = (v) => typeof v === "string" && v.length > 0;

/**
 * Field name -> [predicate, human description]. `climatology` is the only
 * nullable field (schema: `Optional[float]`); it's checked separately below.
 */
const REQUIRED_FIELDS = {
  mode: [isNonEmptyString, "a non-empty string"],
  lat: [isFiniteNumber, "a finite number"],
  lon: [isFiniteNumber, "a finite number"],
  date: [isIsoDate, 'a "YYYY-MM-DD" date'],
  depth: [(v) => isFiniteNumber(v) && v >= 0, "a finite, non-negative number"],
  temperature: [isFiniteNumber, "a finite number"],
  anomaly: [isFiniteNumber, "a finite number"],
  embedding_dim: [(v) => Number.isInteger(v) && v >= 0, "a non-negative integer"],
  data_mode: [isNonEmptyString, "a non-empty string"],
  latency_ms: [(v) => isFiniteNumber(v) && v >= 0, "a finite, non-negative number"],
  cache_hit: [(v) => typeof v === "boolean", "a boolean"],
  notes: [(v) => Array.isArray(v) && v.every((n) => typeof n === "string"), "an array of strings"],
};

/**
 * Validates a `GET /reconstruct` body.
 *
 * @param {*} raw - the parsed JSON body
 * @param {object} [expected]
 * @param {string} [expected.date] - the "YYYY-MM-DD" date that was requested;
 *   when given, the response must echo that same date (a mismatch means the
 *   data belongs to a different timestep than the one asked for).
 * @returns {{ok: true, value: object} | {ok: false, reason: string, fields: string[]}}
 *   `value` is the body itself, unchanged. `fields` lists the offending keys.
 */
export function validatePointReconstruction(raw, { date } = {}) {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    return { ok: false, reason: "GET /reconstruct returned no usable JSON object.", fields: [] };
  }

  const missing = [];
  const invalid = [];
  for (const [field, [check, description]] of Object.entries(REQUIRED_FIELDS)) {
    if (!(field in raw) || raw[field] === undefined || raw[field] === null) {
      missing.push(field);
    } else if (!check(raw[field])) {
      invalid.push(`${field} (expected ${description})`);
    }
  }
  // Nullable, but the key itself must be present and, if not null, a real number.
  if (!("climatology" in raw)) {
    missing.push("climatology");
  } else if (raw.climatology !== null && !isFiniteNumber(raw.climatology)) {
    invalid.push("climatology (expected a finite number or null)");
  }

  if (missing.length > 0) {
    return {
      ok: false,
      reason: `GET /reconstruct response is missing required field(s): ${missing.join(", ")}.`,
      fields: missing,
    };
  }
  if (invalid.length > 0) {
    return {
      ok: false,
      reason: `GET /reconstruct response has invalid field(s): ${invalid.join("; ")}.`,
      fields: invalid.map((s) => s.split(" ")[0]),
    };
  }
  if (date !== undefined && raw.date !== date) {
    return {
      ok: false,
      reason: `GET /reconstruct response is for ${raw.date}, but ${date} was requested.`,
      fields: ["date"],
    };
  }
  return { ok: true, value: raw };
}