// -----------------------------------------------------------------------------
// NEER — North Indian Ocean domain constants  (Phase 34C)
//
// Single source of truth for the spatial domain the map (and, in later
// phases, any other view — vertical-profile, hovmoller, 3d-ocean — that
// needs to reason about the same grid) is drawn against. Mirrors
// `configs/base.yaml`'s `domain:` block on the Python side; if that ever
// changes, update it here too.
//
// Framework-independent (no React) so it can be imported from plain JS,
// tests, or other components without pulling in anything map-specific.
// -----------------------------------------------------------------------------

/** North Indian Ocean reconstruction domain — matches configs/base.yaml. */
export const OCEAN_DOMAIN = Object.freeze({
  latMin: 5,
  latMax: 30,
  lonMin: 45,
  lonMax: 105,
  /** Native grid resolution, in degrees, along both axes. */
  resolution: 0.25,
});

/** Depth levels the reconstruction is produced at, in metres — configs/base.yaml. */
export const DEPTH_LEVELS = Object.freeze([
  0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000,
]);

/** Data-mode vocabulary shared with DataContextPanel. */
export const DATA_MODES = Object.freeze(["reconstructed", "observed", "blended"]);

/**
 * Clamp a value between [min, max].
 */
export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/**
 * True if a lat/lon pair falls within the reconstruction domain (inclusive).
 */
export function isWithinDomain(lat, lon, domain = OCEAN_DOMAIN) {
  return lat >= domain.latMin && lat <= domain.latMax && lon >= domain.lonMin && lon <= domain.lonMax;
}

/**
 * Snap a lat/lon pair to the nearest native grid-cell center at the given
 * resolution (defaults to the domain's own 0.25°), then clamp back into the
 * domain in case snapping pushed it just outside the edge.
 */
export function snapToGrid(lat, lon, domain = OCEAN_DOMAIN) {
  const { resolution } = domain;
  const snappedLat = Math.round(lat / resolution) * resolution;
  const snappedLon = Math.round(lon / resolution) * resolution;
  return {
    lat: clamp(round2(snappedLat), domain.latMin, domain.latMax),
    lon: clamp(round2(snappedLon), domain.lonMin, domain.lonMax),
  };
}

/** Round to 2 decimal places, trimming binary floating-point noise. */
export function round2(value) {
  return Math.round(value * 100) / 100;
}

/** Zero-padded grid index of a coordinate along one axis, from the domain's minimum. */
export function gridIndex(value, min, domain = OCEAN_DOMAIN) {
  return Math.round((value - min) / domain.resolution);
}

/** Format a latitude as e.g. "12.25°N" / "3.00°S". */
export function formatLat(lat) {
  const hemisphere = lat >= 0 ? "N" : "S";
  return `${Math.abs(lat).toFixed(2)}°${hemisphere}`;
}

/** Format a longitude as e.g. "82.50°E" / "12.00°W". */
export function formatLon(lon) {
  const hemisphere = lon >= 0 ? "E" : "W";
  return `${Math.abs(lon).toFixed(2)}°${hemisphere}`;
}

/** Format a depth level in metres, e.g. "0 m (surface)" / "200 m". */
export function formatDepth(depth) {
  if (depth === 0) return "0 m (surface)";
  return `${depth} m`;
}

/**
 * Turns a raw map coordinate into the selection object the dashboard shares.
 *
 * In-domain points snap to the nearest native grid-cell center and carry the
 * zero-based `cell` indices (row from `latMin`, column from `lonMin`, the same
 * axes the backend grid uses). Out-of-domain points keep their rounded raw
 * coordinates with `cell: null` — they are still selectable (so the UI can
 * say "outside the domain") but belong to no grid cell. Non-finite input
 * returns `null`.
 *
 * @returns {{lat: number, lon: number, cell: {latIndex: number, lonIndex: number}|null}|null}
 */
export function toGridSelection(lat, lon, domain = OCEAN_DOMAIN) {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (!isWithinDomain(lat, lon, domain)) return { lat: round2(lat), lon: round2(lon), cell: null };
  const snapped = snapToGrid(lat, lon, domain);
  return {
    ...snapped,
    cell: {
      latIndex: gridIndex(snapped.lat, domain.latMin, domain),
      lonIndex: gridIndex(snapped.lon, domain.lonMin, domain),
    },
  };
}