"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — useOceanMapLayer  (Phase 35A)
//
// Real backend-data plumbing for OceanMap's variable tabs, deliberately kept
// separate from any on-map rendering. Phase 34C's spec explicitly deferred
// "the temperature or anomaly data layers" to a later phase and Phase 35A's
// spec repeats that ("Do not implement the temperature or anomaly data
// layers yet") while asking to "prepare the component architecture for
// backend-driven scientific layers" — this hook IS that architecture: it
// fetches a real grid from GET /reconstruct/grid (frontend/lib/api.js,
// Phase 33B) whenever the active variable has a backend field to back it,
// and exposes the result (or a precise reason it can't fetch one) for
// OceanMap's status line to read. Nothing here draws a heat-map/contour on
// the map; a future phase turns `values` into that.
//
// Variable -> backend support
// ----------------------------
// GET /reconstruct/grid (backend/app/routers/reconstruct.py,
// GridReconstructionResponse in backend/app/schemas.py) returns only
// `temperature` and an optional `climatology` — there is no grid endpoint
// yet for salinity, currents, SSH/SLA, winds, subsurface temperature, or
// uncertainty, and (same "no fake scientific data" rule OceanMap.js itself
// already follows) nothing here invents a stand-in for those. `sst` reads
// `temperature` directly. `anomaly` has no grid field at all — unlike GET
// /reconstruct (the *point* endpoint), which returns anomaly precomputed —
// so it's derived client-side as `temperature - climatology`, the same
// subtraction the backend does for the point endpoint, and is only
// available when the backend actually returned a `climatology` grid for
// that date (it's nullable per the schema).
//
// Phase 35B wires this hook's `values`/`grid` into an actual on-map layer
// (see TemperatureLayer.js) and adds `retry` below so a failed request can
// be retried from an ErrorState without waiting for a param change.
export const LAYER_SUPPORT = Object.freeze({
  sst: { supported: true, field: "temperature" },
  anomaly: { supported: true, field: "anomaly" },
  sss: { supported: false, reason: "No backend grid endpoint yet for salinity" },
  currents: { supported: false, reason: "No backend grid endpoint yet for currents" },
  ssh: { supported: false, reason: "No backend grid endpoint yet for SSH/SLA" },
  winds: { supported: false, reason: "No backend grid endpoint yet for winds" },
  subsurface: { supported: false, reason: "No backend grid endpoint yet for subsurface temperature" },
  uncertainty: { supported: false, reason: "No backend grid endpoint yet for uncertainty" },
});

// Mirrors backend/app/services/repository.py's DEFAULT_MAX_GRID_POINTS —
// GET /reconstruct/grid rejects ("invalid_parameter", 400) a region with
// more cells than this. Checked client-side before ever sending a request,
// rather than sending one the backend is guaranteed to reject; keep this in
// sync if the backend constant changes.
export const MAX_GRID_POINTS = 4096;

import { useCallback, useEffect, useMemo, useRef } from "react";
import { reconstructGrid } from "@/lib/api";
import { useApiRequest } from "@/lib/useApiRequest";
import { OCEAN_DOMAIN } from "@/lib/oceanDomain";

/**
 * Element-wise subtraction of two matching-shape, arbitrarily-nested
 * (2D or 3D) number arrays — `temperature`/`climatology` are `(n_lat,
 * n_lon)` when the caller requested one depth, `(n_lat, n_lon, num_depths)`
 * when it requested every depth level.
 */
function subtractGrids(a, b) {
  if (typeof a === "number") return a - (typeof b === "number" ? b : 0);
  return a.map((item, i) => subtractGrids(item, b[i]));
}

/**
 * @param {object} params
 * @param {string} params.variable - active VARIABLE_TABS value, e.g. "sst"
 * @param {string|Date|null|undefined} params.date
 * @param {number} params.depth - metres
 * @param {{latMin:number, latMax:number, lonMin:number, lonMax:number}} params.bounds
 *   - region to request, typically the map's current visible viewport
 *     clamped to OCEAN_DOMAIN. Panning/zooming naturally brings a request
 *     under the cell limit rather than this hook silently downsampling a
 *     request the backend would reject.
 * @returns {{
 *   supported: boolean,
 *   reason: string|null,
 *   status: "idle"|"loading"|"success"|"error",
 *   isLoading: boolean,
 *   isError: boolean,
 *   error: import("@/lib/api").ApiError|null,
 *   values: (number[][]|number[][][])|null,
 *   temperatureValues: (number[][]|number[][][])|null,
 *   anomalyValues: (number[][]|number[][][])|null,
 *   grid: {lat: number[], lon: number[], depth: number|null, depths: number[]|null}|null,
 *   estimatedCells: number,
 *   tooLargeForCellLimit: boolean,
 * }}
 */
export function useOceanMapLayer({ variable, date, depth, bounds }) {
  const support = LAYER_SUPPORT[variable] ?? { supported: false, reason: "Unknown variable" };
  const { latMin, latMax, lonMin, lonMax } = bounds;

  const estimatedCells = useMemo(() => {
    const latCells = Math.max(1, Math.round((latMax - latMin) / OCEAN_DOMAIN.resolution) + 1);
    const lonCells = Math.max(1, Math.round((lonMax - lonMin) / OCEAN_DOMAIN.resolution) + 1);
    return latCells * lonCells;
  }, [latMin, latMax, lonMin, lonMax]);
  const tooLargeForCellLimit = estimatedCells > MAX_GRID_POINTS;

  const canFetch = support.supported && !tooLargeForCellLimit && Boolean(date);

  const { data, error, status, isLoading, isError, run, reset } = useApiRequest(reconstructGrid);

  // Keyed on the variable *family* (both "sst" and "anomaly" read the same
  // /reconstruct/grid response) rather than the variable itself, so
  // switching between those two tabs never re-fetches — only a change that
  // actually needs a different response does.
  const family = support.supported ? "temperature" : null;
  const requestKey = canFetch
    ? [family, String(date), depth, latMin.toFixed(2), latMax.toFixed(2), lonMin.toFixed(2), lonMax.toFixed(2)].join(
        "|"
      )
    : null;
  const lastKeyRef = useRef(null);

  useEffect(() => {
    if (!canFetch) {
      if (lastKeyRef.current !== null) reset();
      lastKeyRef.current = null;
      return;
    }
    if (lastKeyRef.current === requestKey) return;
    lastKeyRef.current = requestKey;
    run({ latMin, latMax, lonMin, lonMax, date, depth }).catch(() => {
      // Swallowed here on purpose — `error`/`isError` below already
      // capture the failure for the caller to render.
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canFetch, requestKey]);

  const values = useMemo(() => {
    if (!data || !support.supported) return null;
    if (support.field === "temperature") return data.temperature;
    if (support.field === "anomaly") {
      return data.climatology ? subtractGrids(data.temperature, data.climatology) : null;
    }
    return null;
  }, [data, support]);

  // Phase 35C-B: the hover tooltip (OceanMap.js) shows temperature *and*
  // anomaly together, regardless of which single variable tab is active —
  // both read from this one already-fetched response (never a second
  // request) whenever it actually contains them, same "never invent a
  // scientific value" rule `values` above already follows: anomaly stays
  // `null` here exactly when it would have above (no `climatology` for
  // this date), rather than silently falling back to something else.
  const temperatureValues = useMemo(() => {
    if (!data || !support.supported) return null;
    return data.temperature ?? null;
  }, [data, support]);

  const anomalyValues = useMemo(() => {
    if (!data || !support.supported) return null;
    return data.climatology ? subtractGrids(data.temperature, data.climatology) : null;
  }, [data, support]);

  const grid = data ? { lat: data.lat, lon: data.lon, depth: data.depth, depths: data.depths } : null;

  // Re-issues the same request after a failure (e.g. a transient network
  // error) without waiting for date/depth/bounds to change — the request
  // key above only re-fires the effect on a genuine parameter change, so a
  // caller-triggered retry after `isError` needs its own escape hatch.
  const retry = useCallback(() => {
    if (!canFetch) return;
    run({ latMin, latMax, lonMin, lonMax, date, depth }).catch(() => {
      // Swallowed here on purpose — `error`/`isError` above already
      // capture the failure for the caller to render.
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canFetch, latMin, latMax, lonMin, lonMax, date, depth]);

  return {
    supported: support.supported,
    reason: support.supported ? null : support.reason,
    status,
    isLoading,
    isError,
    error,
    values,
    temperatureValues,
    anomalyValues,
    grid,
    estimatedCells,
    tooLargeForCellLimit,
    retry,
  };
}

export default useOceanMapLayer;