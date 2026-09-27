// -----------------------------------------------------------------------------
// NEER — continuous color-scale utilities  (Phase 35B)
//
// Framework-independent (no React), same convention as oceanDomain.js, so it
// can be reused by any scientific layer/legend (OceanMap's temperature layer
// today; vertical-profile/hovmoller color-coding later) without pulling in
// anything map-specific. Nothing here invents scientific values — callers
// pass in the real backend-derived numbers; this module only maps a number
// to a color and reports the finite min/max actually present in that data.
// -----------------------------------------------------------------------------

/** Linearly interpolate between two RGB triples at t in [0, 1]. */
function lerpRgb([r1, g1, b1], [r2, g2, b2], t) {
  return [r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t];
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  const n = parseInt(value, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToCss([r, g, b], alpha = 1) {
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
}

/**
 * Samples a piecewise-linear color scale defined by [t, hexColor] stops.
 * `t` outside [stops[0][0], stops[last][0]] is clamped to the nearest end.
 */
function sampleStops(stops, t) {
  const clamped = Math.min(stops[stops.length - 1][0], Math.max(stops[0][0], t));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const local = t1 === t0 ? 0 : (clamped - t0) / (t1 - t0);
      return lerpRgb(hexToRgb(c0), hexToRgb(c1), local);
    }
  }
  return hexToRgb(stops[stops.length - 1][1]);
}

// Sequential "thermal" scale — cold (deep blue) through neutral to hot (deep
// red), the conventional palette for absolute sea-surface temperature. t in
// [0, 1], where 0 is the coldest value actually present in the field and 1
// the warmest.
const THERMAL_STOPS = [
  [0.0, "#08306B"],
  [0.2, "#2171B5"],
  [0.4, "#6BAED6"],
  [0.55, "#FFFFBF"],
  [0.7, "#FDAE61"],
  [0.85, "#F46D43"],
  [1.0, "#A50026"],
];

// Diverging scale — cool blue for negative, white/neutral at zero, warm red
// for positive; the conventional palette for a signed anomaly field. t in
// [-1, 1].
const DIVERGING_STOPS = [
  [-1.0, "#2166AC"],
  [-0.5, "#67A9CF"],
  [-0.1, "#D1E5F0"],
  [0.0, "#F7F7F7"],
  [0.1, "#FDDBC7"],
  [0.5, "#EF8A62"],
  [1.0, "#B2182B"],
];

/**
 * @param {number} t - normalized position, 0 (coldest) to 1 (warmest)
 * @param {number} [alpha]
 * @returns {string} CSS rgba(...) color
 */
export function thermalColor(t, alpha = 1) {
  return rgbToCss(sampleStops(THERMAL_STOPS, t), alpha);
}

/**
 * @param {number} t - normalized position, -1 (most negative) to 1 (most positive)
 * @param {number} [alpha]
 * @returns {string} CSS rgba(...) color
 */
export function divergingColor(t, alpha = 1) {
  return rgbToCss(sampleStops(DIVERGING_STOPS, t), alpha);
}

/**
 * Finite-value min/max of an arbitrarily-nested numeric array, treating
 * `null`/`undefined`/`NaN` cells (the backend's convention for a masked or
 * invalid grid cell — see useOceanMapLayer.js) as missing rather than as 0
 * or a scale endpoint. Returns `null` if the field has no finite values at
 * all (e.g. an entirely masked/land-only region).
 * @param {*} nested
 * @returns {{min: number, max: number}|null}
 */
export function computeFiniteExtent(nested) {
  let min = Infinity;
  let max = -Infinity;
  function visit(node) {
    if (Array.isArray(node)) {
      node.forEach(visit);
      return;
    }
    if (typeof node === "number" && Number.isFinite(node)) {
      if (node < min) min = node;
      if (node > max) max = node;
    }
  }
  visit(nested);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  return { min, max };
}

/**
 * Builds a `value -> CSS color` function for one loaded field, choosing a
 * sequential or diverging scale by `kind` and normalizing against the
 * field's own finite extent (never a hard-coded physical range).
 *
 * @param {"sequential"|"diverging"} kind
 * @param {{min: number, max: number}} extent - real extent from computeFiniteExtent
 * @returns {(value: number|null|undefined) => (string|null)} null for a
 *   missing/invalid value — callers should skip drawing that cell rather
 *   than rendering a fabricated color for it
 */
export function makeColorMapper(kind, extent) {
  if (kind === "diverging") {
    const maxAbs = Math.max(Math.abs(extent.min), Math.abs(extent.max)) || 1;
    return (value) => {
      if (typeof value !== "number" || !Number.isFinite(value)) return null;
      return divergingColor(Math.max(-1, Math.min(1, value / maxAbs)));
    };
  }
  const span = extent.max - extent.min || 1;
  return (value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return thermalColor((value - extent.min) / span);
  };
}

/** Color-scale kind + a display unit for one OceanMap variable. */
export const VARIABLE_COLOR_CONFIG = Object.freeze({
  sst: { kind: "sequential", unit: "°C" },
  anomaly: { kind: "diverging", unit: "°C" },
});