// -----------------------------------------------------------------------------
// NEER Frontend — date/depth state model  (Phase 36A)
//
// The single, framework-independent definition of what "the available dates,
// the selected date, the available depths, and the selected depth" mean for
// the whole app, and of every rule for keeping them consistent:
//
//   - payload parsing/validation for GET /dates and GET /model/info
//   - comparison of the backend depth list against the expected NEER model
//     configuration
//   - a reducer (`dateDepthReducer`) owning the state and every transition
//   - selectors/helpers consumers use instead of re-deriving things
//
// No React, no fetch, no import of api.js (api.js imports *this* file to
// validate responses — importing it back would be a cycle). Everything here
// is pure so it runs under the repo's node-only vitest setup; the React layer
// (lib/useDateDepth.js) is just a thin wrapper that feeds API results into
// the reducer.
//
// REPRESENTATION (the "consistent everywhere" requirement)
//   - a date is ALWAYS an ISO calendar-date string, "YYYY-MM-DD" — the exact
//     format the backend's /dates returns and its query params accept. Never
//     a Date object inside state: a Date drags in time zones (a UTC-midnight
//     Date renders as the previous day in the Americas), a string doesn't.
//   - a depth is ALWAYS a finite number of metres, and ALWAYS one of the
//     entries in `availableDepths` (the backend snaps unknown depths to the
//     nearest level; the frontend state never holds an off-list depth, so
//     what the UI shows is exactly what was requested).
//   - `null` means "nothing valid is selected (yet)" — never a placeholder
//     default. The old hard-coded "2020-01-01" default is gone: a date is
//     only selectable if the backend said it exists.
// -----------------------------------------------------------------------------

import { DEPTH_LEVELS } from "./oceanDomain.js";

/**
 * Expected NEER model depth configuration (metres) — configs/base.yaml, and
 * the list the phase spec fixes. Used to *validate* the backend's list, and
 * as an explicitly-labelled fallback only when the backend list can't be
 * obtained (see `depthsFailed`). Never used to silently override a valid
 * backend-provided list.
 */
export const EXPECTED_DEPTHS = DEPTH_LEVELS;

/** Load status shared by dates and depths. */
export const LOAD_STATUS = Object.freeze({
  LOADING: "loading",
  SUCCESS: "success",
  /** Request succeeded but the backend reported zero dates. Not an error. */
  EMPTY: "empty",
  ERROR: "error",
});

/** Where the current depth list came from. */
export const DEPTH_SOURCE = Object.freeze({
  /** Read from GET /model/info (`architecture.depths`). */
  BACKEND: "backend",
  /** Backend list unavailable/invalid — the expected NEER list, clearly labelled. */
  EXPECTED_FALLBACK: "expected_fallback",
});

// ---------------------------------------------------------------------------
// Dates
// ---------------------------------------------------------------------------

const ISO_DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/;

/**
 * True only for a real calendar date in strict "YYYY-MM-DD" form —
 * "2020-02-30" matches the pattern but is rejected by the round-trip check.
 * @param {*} value
 * @returns {boolean}
 */
export function isIsoDate(value) {
  if (typeof value !== "string") return false;
  const match = ISO_DATE_PATTERN.exec(value);
  if (!match) return false;
  const [, y, m, d] = match.map(Number);
  const utc = new Date(Date.UTC(y, m - 1, d));
  return utc.getUTCFullYear() === y && utc.getUTCMonth() === m - 1 && utc.getUTCDate() === d;
}

/**
 * Normalizes a date input to the canonical "YYYY-MM-DD" string, or `null` if
 * it isn't a valid date. A `Date` is read in UTC — the same convention
 * api.js's `toDateParam` uses when it sends one to the backend.
 * @param {string|Date|null|undefined} value
 * @returns {string|null}
 */
export function toIsoDate(value) {
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value.toISOString().slice(0, 10);
  }
  return isIsoDate(value) ? value : null;
}

/**
 * Parses and validates a GET /dates body (backend `DatesResponse`).
 *
 * Invalid (`ok: false`): not an object, `dates` not an array, or any entry
 * that isn't a real "YYYY-MM-DD" date. Discrepancies that don't make the
 * list itself untrustworthy — duplicates, a `count` that disagrees with the
 * list, `min_date`/`max_date` that disagree with it — are reported in
 * `warnings` instead; the *list* is the source of truth and is what's
 * returned (de-duplicated, ascending).
 *
 * An empty `dates` array is VALID here (it's the empty state, not an error).
 *
 * @param {*} raw
 * @returns {{ok: true, value: {dates: string[], count: number, minDate: string|null,
 *   maxDate: string|null, dataMode: string, isSynthetic: boolean, warnings: string[]}}
 *   | {ok: false, reason: string}}
 */
export function parseDatesPayload(raw) {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    return { ok: false, reason: "GET /dates returned no usable JSON object." };
  }
  if (!Array.isArray(raw.dates)) {
    return { ok: false, reason: "GET /dates response is missing a `dates` array." };
  }
  const badIndex = raw.dates.findIndex((d) => !isIsoDate(d));
  if (badIndex !== -1) {
    return {
      ok: false,
      reason: `GET /dates entry #${badIndex} is not a valid YYYY-MM-DD date: ${JSON.stringify(raw.dates[badIndex])}.`,
    };
  }

  const warnings = [];
  const unique = [...new Set(raw.dates)].sort(); // ISO strings sort chronologically
  if (unique.length !== raw.dates.length) {
    warnings.push(`GET /dates contained ${raw.dates.length - unique.length} duplicate date(s); they were removed.`);
  }
  if (typeof raw.count === "number" && raw.count !== unique.length) {
    warnings.push(`GET /dates reported count=${raw.count} but listed ${unique.length} distinct date(s).`);
  }
  const minDate = unique.length ? unique[0] : null;
  const maxDate = unique.length ? unique[unique.length - 1] : null;
  if (raw.min_date != null && raw.min_date !== minDate) {
    warnings.push(`GET /dates min_date (${raw.min_date}) does not match the earliest listed date (${minDate}).`);
  }
  if (raw.max_date != null && raw.max_date !== maxDate) {
    warnings.push(`GET /dates max_date (${raw.max_date}) does not match the latest listed date (${maxDate}).`);
  }

  return {
    ok: true,
    value: {
      dates: unique,
      count: unique.length,
      minDate,
      maxDate,
      dataMode: typeof raw.data_mode === "string" ? raw.data_mode : "UNKNOWN",
      isSynthetic: raw.is_synthetic === true,
      warnings,
    },
  };
}

/**
 * Default selection for a date list: the latest available date (the most
 * recent field is what an operational dashboard should open on), or `null`
 * for an empty list. A current selection that's still in the list wins.
 * @param {string[]} dates - ascending, de-duplicated
 * @param {string|null} current
 * @returns {string|null}
 */
export function reconcileDate(dates, current) {
  if (!dates.length) return null;
  if (current && dates.includes(current)) return current;
  return dates[dates.length - 1];
}

// ---------------------------------------------------------------------------
// Depths
// ---------------------------------------------------------------------------

/**
 * Validates a depth list: non-empty array of finite, non-negative numbers,
 * strictly increasing (no duplicates, no reordering — a depth axis that
 * isn't monotonic isn't a depth axis).
 * @param {*} list
 * @returns {{ok: true, depths: number[]} | {ok: false, reason: string}}
 */
export function validateDepthList(list) {
  if (!Array.isArray(list) || list.length === 0) {
    return { ok: false, reason: "depth list is missing or empty." };
  }
  for (let i = 0; i < list.length; i += 1) {
    const d = list[i];
    if (typeof d !== "number" || !Number.isFinite(d) || d < 0) {
      return { ok: false, reason: `depth #${i} is not a finite, non-negative number: ${JSON.stringify(d)}.` };
    }
    if (i > 0 && d <= list[i - 1]) {
      return { ok: false, reason: `depth list is not strictly increasing at #${i} (${list[i - 1]} → ${d}).` };
    }
  }
  return { ok: true, depths: [...list] };
}

/**
 * Compares a depth list against `EXPECTED_DEPTHS`.
 * @param {number[]} depths
 * @returns {{matches: boolean, missing: number[], unexpected: number[]}}
 *   `missing` = expected levels the list lacks; `unexpected` = extra levels.
 */
export function compareWithExpectedDepths(depths) {
  const have = new Set(depths);
  const expected = new Set(EXPECTED_DEPTHS);
  const missing = EXPECTED_DEPTHS.filter((d) => !have.has(d));
  const unexpected = depths.filter((d) => !expected.has(d));
  return { matches: missing.length === 0 && unexpected.length === 0, missing, unexpected };
}

/**
 * Extracts and validates the depth configuration from a GET /model/info
 * body (`architecture.depths` / `architecture.num_depths`).
 *
 * Invalid (`ok: false`): body/architecture not objects, or the depth list
 * fails `validateDepthList`, or `num_depths` contradicts the list's length.
 *
 * A *valid* list that differs from the expected NEER configuration is still
 * `ok: true` — the backend's list is authoritative for what the loaded model
 * actually supports (per the phase spec, never silently replaced) — but it's
 * reported via `matchesExpected: false` + `missing`/`unexpected` + a
 * `warnings` entry so the discrepancy is visible.
 *
 * @param {*} raw
 * @returns {{ok: true, value: {depths: number[], matchesExpected: boolean,
 *   missing: number[], unexpected: number[], warnings: string[]}}
 *   | {ok: false, reason: string}}
 */
export function parseModelInfoDepths(raw) {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    return { ok: false, reason: "GET /model/info returned no usable JSON object." };
  }
  const arch = raw.architecture;
  if (arch === null || typeof arch !== "object" || Array.isArray(arch)) {
    return { ok: false, reason: "GET /model/info response is missing `architecture`." };
  }
  const checked = validateDepthList(arch.depths);
  if (!checked.ok) {
    return { ok: false, reason: `GET /model/info architecture.depths: ${checked.reason}` };
  }
  if (typeof arch.num_depths === "number" && arch.num_depths !== checked.depths.length) {
    return {
      ok: false,
      reason: `GET /model/info reports num_depths=${arch.num_depths} but lists ${checked.depths.length} depth(s).`,
    };
  }

  const cmp = compareWithExpectedDepths(checked.depths);
  const warnings = [];
  if (!cmp.matches) {
    const parts = [];
    if (cmp.missing.length) parts.push(`missing expected level(s) ${cmp.missing.join(", ")}`);
    if (cmp.unexpected.length) parts.push(`unexpected level(s) ${cmp.unexpected.join(", ")}`);
    warnings.push(`Backend depth list differs from the expected NEER configuration: ${parts.join("; ")}.`);
  }
  return {
    ok: true,
    value: {
      depths: checked.depths,
      matchesExpected: cmp.matches,
      missing: cmp.missing,
      unexpected: cmp.unexpected,
      warnings,
    },
  };
}

/**
 * Default selection for a depth list: keep the current depth if it's still
 * available, else the surface (0 m) if present, else the shallowest level;
 * `null` for an empty list.
 * @param {number[]} depths
 * @param {number|null} current
 * @returns {number|null}
 */
export function reconcileDepth(depths, current) {
  if (!depths.length) return null;
  if (current !== null && current !== undefined && depths.includes(current)) return current;
  return depths.includes(0) ? 0 : depths[0];
}

// ---------------------------------------------------------------------------
// State + reducer
// ---------------------------------------------------------------------------

/**
 * @typedef {object} DatesState
 * @property {"loading"|"success"|"empty"|"error"} status
 * @property {string[]} items - ascending ISO dates; [] unless status is "success"
 * @property {string|null} minDate
 * @property {string|null} maxDate
 * @property {string|null} dataMode - backend `data_mode`
 * @property {boolean} isSynthetic
 * @property {string[]} warnings
 * @property {object|null} error - the ApiError (status "error" only)
 * @property {string|null} errorCategory - see api.js `categorizeApiError`
 *
 * @typedef {object} DepthsState
 * @property {"loading"|"success"|"error"} status - "error" means the backend
 *   list couldn't be obtained/validated; `items` is then the expected-list
 *   fallback and `source` says so
 * @property {number[]} items - ascending metres
 * @property {"backend"|"expected_fallback"|null} source
 * @property {boolean|null} matchesExpected - null until known
 * @property {number[]} missing
 * @property {number[]} unexpected
 * @property {string[]} warnings
 * @property {object|null} error
 * @property {string|null} errorCategory
 *
 * @typedef {object} DateDepthState
 * @property {DatesState} dates
 * @property {DepthsState} depths
 * @property {string|null} selectedDate
 * @property {number|null} selectedDepth
 */

/** @returns {DateDepthState} */
export function createInitialState() {
  return {
    dates: {
      status: LOAD_STATUS.LOADING,
      items: [],
      minDate: null,
      maxDate: null,
      dataMode: null,
      isSynthetic: false,
      warnings: [],
      error: null,
      errorCategory: null,
    },
    depths: {
      status: LOAD_STATUS.LOADING,
      items: [],
      source: null,
      matchesExpected: null,
      missing: [],
      unexpected: [],
      warnings: [],
      error: null,
      errorCategory: null,
    },
    selectedDate: null,
    selectedDepth: null,
  };
}

/** Action creators — the only way components/hooks should talk to the reducer. */
export const dateDepthActions = {
  datesLoading: () => ({ type: "dates/loading" }),
  /** @param {*} parsed - the `value` of a successful `parseDatesPayload` */
  datesLoaded: (parsed) => ({ type: "dates/loaded", parsed }),
  datesFailed: (error, category) => ({ type: "dates/failed", error, category }),
  depthsLoading: () => ({ type: "depths/loading" }),
  /** @param {*} parsed - the `value` of a successful `parseModelInfoDepths` */
  depthsLoaded: (parsed) => ({ type: "depths/loaded", parsed }),
  depthsFailed: (error, category) => ({ type: "depths/failed", error, category }),
  selectDate: (date) => ({ type: "date/select", date }),
  /** @param {number} delta - +1 = next available date, -1 = previous */
  stepDate: (delta) => ({ type: "date/step", delta }),
  selectDepth: (depth) => ({ type: "depth/select", depth }),
};

/**
 * @param {DateDepthState} state
 * @param {{type: string}} action
 * @returns {DateDepthState}
 */
export function dateDepthReducer(state, action) {
  switch (action.type) {
    case "dates/loading":
      return { ...state, dates: { ...state.dates, status: LOAD_STATUS.LOADING, error: null, errorCategory: null } };

    case "dates/loaded": {
      const p = action.parsed;
      return {
        ...state,
        dates: {
          status: p.dates.length ? LOAD_STATUS.SUCCESS : LOAD_STATUS.EMPTY,
          items: p.dates,
          minDate: p.minDate,
          maxDate: p.maxDate,
          dataMode: p.dataMode,
          isSynthetic: p.isSynthetic,
          warnings: p.warnings,
          error: null,
          errorCategory: null,
        },
        selectedDate: reconcileDate(p.dates, state.selectedDate),
      };
    }

    case "dates/failed":
      // Never keep showing dates we can't currently vouch for.
      return {
        ...state,
        dates: {
          ...createInitialState().dates,
          status: LOAD_STATUS.ERROR,
          error: action.error,
          errorCategory: action.category ?? null,
        },
        selectedDate: null,
      };

    case "depths/loading":
      return { ...state, depths: { ...state.depths, status: LOAD_STATUS.LOADING, error: null, errorCategory: null } };

    case "depths/loaded": {
      const p = action.parsed;
      return {
        ...state,
        depths: {
          status: LOAD_STATUS.SUCCESS,
          items: p.depths,
          source: DEPTH_SOURCE.BACKEND,
          matchesExpected: p.matchesExpected,
          missing: p.missing,
          unexpected: p.unexpected,
          warnings: p.warnings,
          error: null,
          errorCategory: null,
        },
        selectedDepth: reconcileDepth(p.depths, state.selectedDepth),
      };
    }

    case "depths/failed": {
      // Backend list unobtainable or invalid: fall back to the expected NEER
      // list, but say so — `status: "error"` + `source: "expected_fallback"`
      // + the error itself stay on the state for the UI to surface. This is
      // the one place the frontend supplies a depth list of its own.
      const items = [...EXPECTED_DEPTHS];
      return {
        ...state,
        depths: {
          status: LOAD_STATUS.ERROR,
          items,
          source: DEPTH_SOURCE.EXPECTED_FALLBACK,
          matchesExpected: null,
          missing: [],
          unexpected: [],
          warnings: [],
          error: action.error,
          errorCategory: action.category ?? null,
        },
        selectedDepth: reconcileDepth(items, state.selectedDepth),
      };
    }

    case "date/select": {
      const iso = toIsoDate(action.date);
      // Only a date the backend actually listed can become the selection.
      if (!iso || !state.dates.items.includes(iso) || iso === state.selectedDate) return state;
      return { ...state, selectedDate: iso };
    }

    case "date/step": {
      const { items } = state.dates;
      const i = state.selectedDate ? items.indexOf(state.selectedDate) : -1;
      if (i === -1) return state;
      const next = items[i + Math.sign(action.delta)];
      return next === undefined ? state : { ...state, selectedDate: next };
    }

    case "depth/select": {
      if (!state.depths.items.includes(action.depth) || action.depth === state.selectedDepth) return state;
      return { ...state, selectedDepth: action.depth };
    }

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

/**
 * Whether stepping by `delta` available dates is possible from the current
 * selection (drives enabled/disabled state of prev/next controls).
 * @param {DateDepthState} state
 * @param {number} delta
 * @returns {boolean}
 */
export function canStepDate(state, delta) {
  const { items } = state.dates;
  const i = state.selectedDate ? items.indexOf(state.selectedDate) : -1;
  return i !== -1 && items[i + Math.sign(delta)] !== undefined;
}

/**
 * True when both a valid date and a valid depth are selected — i.e. it is
 * safe to issue a query that takes them (reconstruct, grid, profile, ...).
 * @param {DateDepthState} state
 * @returns {boolean}
 */
export function isSelectionReady(state) {
  return state.selectedDate !== null && state.selectedDepth !== null;
}

/**
 * Display options for a depth selector: `[{label, value}]`, in the shape
 * DataContextPanel's `depthLevels` prop already uses.
 * @param {number[]} depths
 * @returns {{label: string, value: number}[]}
 */
export function depthOptions(depths) {
  return depths.map((value) => ({ value, label: value === 0 ? "Surface" : `${value} m` }));
}