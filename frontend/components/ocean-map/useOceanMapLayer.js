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
// GET /reconstruct/grid returns aligned `temperature`, `climatology`, and
// canonical `anomaly` fields. The model anomaly is pooled by depth; the
// backend returns it on the same coordinate grid and masks cells without a
// valid climatology. The frontend validates and consumes that field directly.
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
import { validateGridAnomalyResponse } from "@/lib/anomaly";

/** Estimated number of native-resolution cells a bounds box covers. */
export function estimateCells({ latMin, latMax, lonMin, lonMax }) {
  const latCells = Math.max(1, Math.round((latMax - latMin) / OCEAN_DOMAIN.resolution) + 1);
  const lonCells = Math.max(1, Math.round((lonMax - lonMin) / OCEAN_DOMAIN.resolution) + 1);
  return latCells * lonCells;
}

/** True when `outer` fully covers `inner` (both {latMin,latMax,lonMin,lonMax}). */
export function boundsContain(outer, inner) {
  if (!outer || !inner) return false;
  const eps = 1e-9;
  return (
    outer.latMin <= inner.latMin + eps &&
    outer.latMax >= inner.latMax - eps &&
    outer.lonMin <= inner.lonMin + eps &&
    outer.lonMax >= inner.lonMax - eps
  );
}

/**
 * Bounds actually sent to GET /reconstruct/grid for a given visible viewport.
 *
 * The viewport is snapped OUTWARD to whole degrees (and clamped to the
 * domain), so a small pan usually resolves to the same request — same cache
 * key, no new API call — instead of a fresh request per pixel of drag. It
 * only ever grows the region, never shrinks it, so everything visible is
 * still covered. If snapping would push the request past the backend's cell
 * limit, the exact viewport is used instead (never a request the backend is
 * guaranteed to reject).
 */
export function planRequestBounds(bounds) {
  const snapped = {
    latMin: Math.max(OCEAN_DOMAIN.latMin, Math.floor(bounds.latMin)),
    latMax: Math.min(OCEAN_DOMAIN.latMax, Math.ceil(bounds.latMax)),
    lonMin: Math.max(OCEAN_DOMAIN.lonMin, Math.floor(bounds.lonMin)),
    lonMax: Math.min(OCEAN_DOMAIN.lonMax, Math.ceil(bounds.lonMax)),
  };
  return estimateCells(snapped) <= MAX_GRID_POINTS ? snapped : bounds;
}

// ---------------------------------------------------------------------------
// Phase 36B — date/depth-driven requests
// ---------------------------------------------------------------------------

/** Most recent grid responses kept so stepping back to a date/depth just viewed is instant. */
export const GRID_CACHE_LIMIT = 12;

/**
 * A depth is only requestable when it is a real, finite, non-negative number.
 * `reconstructGrid` treats a missing depth as "return EVERY depth level" (a 3D
 * array this 2D map can't draw), so a not-yet-known depth (null before GET
 * /model/info answers) must never be sent as "no depth".
 */
export function isRequestableDepth(depth) {
  return typeof depth === "number" && Number.isFinite(depth) && depth >= 0;
}

/** Identity of one GET /reconstruct/grid response: date + depth + region. */
export function gridCacheKey({ date, depth, latMin, latMax, lonMin, lonMax }) {
  return [String(date), depth, latMin.toFixed(2), latMax.toFixed(2), lonMin.toFixed(2), lonMax.toFixed(2)].join("|");
}

/** True when a loaded field belongs to exactly the requested date and depth. */
export function fieldContextMatches(context, { date, depth, bounds }) {
  const coordinatesMatch = !bounds || Boolean(context?.bounds
    && context.bounds.latMin <= bounds.latMin && context.bounds.latMax >= bounds.latMax
    && context.bounds.lonMin <= bounds.lonMin && context.bounds.lonMax >= bounds.lonMax);
  return Boolean(context) && context.date === String(date) && context.depth === depth && coordinatesMatch;
}

/** Inserts into an insertion-ordered Map used as a small LRU. */
export function rememberInCache(cache, key, value, limit = GRID_CACHE_LIMIT) {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > limit) cache.delete(cache.keys().next().value);
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
 *   fieldContext: {date: string, depth: number}|null,  // date/depth the drawn field belongs to
 *   isStale: boolean,     // a field is held but it is for a different date/depth than requested
 *   isUpdating: boolean,  // a new date/depth field is being fetched (previous one may still be drawn)
 * }}
 */
const UNKNOWN_VARIABLE = Object.freeze({ supported: false, reason: "Unknown variable" });

export function useOceanMapLayer({ variable, date, depth, bounds }) {
  const support = LAYER_SUPPORT[variable] ?? UNKNOWN_VARIABLE;
  const isSupported = support.supported;
  const fieldName = support.field;
  const viewLatMin = bounds.latMin;
  const viewLatMax = bounds.latMax;
  const viewLonMin = bounds.lonMin;
  const viewLonMax = bounds.lonMax;

  const estimatedCells = useMemo(
    () => estimateCells({ latMin: viewLatMin, latMax: viewLatMax, lonMin: viewLonMin, lonMax: viewLonMax }),
    [viewLatMin, viewLatMax, viewLonMin, viewLonMax]
  );
  const tooLargeForCellLimit = estimatedCells > MAX_GRID_POINTS;

  const hasValidBounds = viewLatMin < viewLatMax && viewLonMin < viewLonMax;

  // Region actually requested — the viewport snapped outward to whole degrees
  // (see planRequestBounds), so nearby pans share one request.
  const { latMin, latMax, lonMin, lonMax } = useMemo(
    () =>
      hasValidBounds
        ? planRequestBounds({ latMin: viewLatMin, latMax: viewLatMax, lonMin: viewLonMin, lonMax: viewLonMax })
        : { latMin: viewLatMin, latMax: viewLatMax, lonMin: viewLonMin, lonMax: viewLonMax },
    [hasValidBounds, viewLatMin, viewLatMax, viewLonMin, viewLonMax]
  );
  const canFetch =
    isSupported && hasValidBounds && !tooLargeForCellLimit && Boolean(date) && isRequestableDepth(depth);

  // Every grid request goes through frontend/lib/api.js's reconstructGrid. The
  // wrapper adds two things, neither of which invents data:
  //   - a small LRU of real responses, so returning to a date/depth/region
  //     already fetched (previous <-> next, chip back and forth) costs no request;
  //   - a record of which date/depth each response belongs to (WeakMap keyed by
  //     the response object), so the hook can tell when the field on screen is
  //     for a DIFFERENT selection than the one now requested ("stale").
  const cacheRef = useRef(new Map());
  const contextRef = useRef(new WeakMap());
  const fetchGrid = useCallback(async (params) => {
    const key = gridCacheKey(params);
    const cached = cacheRef.current.get(key);
    if (cached) {
      rememberInCache(cacheRef.current, key, cached);
      return cached;
    }
    const response = await reconstructGrid(params);
    if (response && typeof response === "object") {
      const validation = validateGridAnomalyResponse(response, params);
      if (!validation.ok) throw new Error(validation.reason);
      contextRef.current.set(response, {
        date: String(params.date), depth: params.depth,
        bounds: { latMin: params.latMin, latMax: params.latMax, lonMin: params.lonMin, lonMax: params.lonMax },
      });
      rememberInCache(cacheRef.current, key, response);
    }
    return response;
  }, []);

  const { data, error, status, isLoading, isError, run, reset } = useApiRequest(fetchGrid);

  // Keyed on the variable *family* (both "sst" and "anomaly" read the same
  // /reconstruct/grid response) rather than the variable itself, so
  // switching between those two tabs never re-fetches — only a change that
  // actually needs a different response does.
  const family = isSupported ? "temperature" : null;
  const requestKey = canFetch
    ? [family, String(date), depth, latMin.toFixed(2), latMax.toFixed(2), lonMin.toFixed(2), lonMax.toFixed(2)].join(
        "|"
      )
    : null;
  const lastKeyRef = useRef(null);
  // What the last issued request covers (params signature + region), and the
  // latest status — refs, not effect deps, so reading them never re-fires the
  // effect. Together they let a zoom-in / small pan that is still fully
  // inside an already-loaded region (same field, date and depth) reuse that
  // data instead of issuing another request for cells we already hold.
  const loadedRef = useRef(null);
  const statusRef = useRef(status);
  statusRef.current = status;
  const paramsSig = canFetch ? [family, String(date), depth].join("|") : null;

  useEffect(() => {
    if (!canFetch) {
      if (lastKeyRef.current !== null) reset();
      lastKeyRef.current = null;
      loadedRef.current = null;
      return;
    }
    if (lastKeyRef.current === requestKey) return;
    if (
      loadedRef.current &&
      loadedRef.current.sig === paramsSig &&
      (statusRef.current === "success" || statusRef.current === "loading") &&
      boundsContain(loadedRef.current.bounds, { latMin, latMax, lonMin, lonMax })
    ) {
      lastKeyRef.current = requestKey;
      return;
    }

    // Debounce rapid changes (slider drag, key repeat) so only the value the
    // user settles on is requested — but a response we already hold needs no
    // waiting.
    const delay = cacheRef.current.has(gridCacheKey({ date, depth, latMin, latMax, lonMin, lonMax })) ? 0 : 200;
    const timer = setTimeout(() => {
      lastKeyRef.current = requestKey;
      loadedRef.current = { sig: paramsSig, bounds: { latMin, latMax, lonMin, lonMax } };
      run({ latMin, latMax, lonMin, lonMax, date, depth }).catch(() => {
        // Swallowed here on purpose — `error`/`isError` below already
        // capture the failure for the caller to render.
      });
    }, delay);

    return () => {
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canFetch, requestKey]);

  // Which date/depth the response currently held is for, and whether that
  // differs from what is now selected.
  //   - while the new field is on its way, the previous valid field stays
  //     drawn (`isUpdating`; the map dims it and says which date/depth it is);
  //   - if the new field FAILED, the old one is dropped: showing another
  //     date's/depth's temperatures under the newly selected label would be
  //     wrong, and the error state explains why the map is blank.
  const fieldContext = data ? (contextRef.current.get(data) ?? null) : null;
  const isStale = Boolean(data) && !fieldContextMatches(fieldContext, { date, depth, bounds: { latMin, latMax, lonMin, lonMax } });
  const keepPreviousField = isStale && canFetch && !isError;
  const fieldData = data && isSupported && (!isStale || keepPreviousField) ? data : null;
  const isUpdating = canFetch && (isLoading || (isStale && !isError));

  const values = useMemo(() => {
    const data = fieldData;
    if (!data || !isSupported) return null;
    if (fieldName === "temperature") return data.temperature;
    if (fieldName === "anomaly") return data.anomaly ?? null;
    return null;
  }, [fieldData, isSupported, fieldName]);

  // Phase 35C-B: the hover tooltip (OceanMap.js) shows temperature *and*
  // anomaly together, regardless of which single variable tab is active —
  // both read from this one already-fetched response (never a second
  // request) whenever it actually contains them, same "never invent a
  // scientific value" rule `values` above already follows: anomaly stays
  // `null` here exactly when it would have above (no `climatology` for
  // this date), rather than silently falling back to something else.
  const temperatureValues = useMemo(() => {
    if (!fieldData || !isSupported) return null;
    return fieldData.temperature ?? null;
  }, [fieldData, isSupported]);

  const anomalyValues = useMemo(() => {
    if (!fieldData || !isSupported) return null;
    return fieldData.anomaly ?? null;
  }, [fieldData, isSupported]);

  const climatologyValues = useMemo(() => {
    if (!fieldData || !isSupported) return null;
    return fieldData.climatology ?? null;
  }, [fieldData, isSupported]);

  // Memoized: a new object identity per render would defeat the memoization
  // of every consumer (TemperatureLayer re-derives all its cells from `grid`).
  const grid = useMemo(
    () => (fieldData ? { lat: fieldData.lat, lon: fieldData.lon, depth: fieldData.depth, depths: fieldData.depths } : null),
    [fieldData]
  );

  // Re-issues the same request after a failure (e.g. a transient network
  // error) without waiting for date/depth/bounds to change — the request
  // key above only re-fires the effect on a genuine parameter change, so a
  // caller-triggered retry after `isError` needs its own escape hatch.
  const retry = useCallback(() => {
    if (!canFetch) return;
    loadedRef.current = { sig: paramsSig, bounds: { latMin, latMax, lonMin, lonMax } };
    run({ latMin, latMax, lonMin, lonMax, date, depth }).catch(() => {
      // Swallowed here on purpose — `error`/`isError` above already
      // capture the failure for the caller to render.
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canFetch, paramsSig, latMin, latMax, lonMin, lonMax, date, depth]);

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
    climatologyValues,
    grid,
    estimatedCells,
    tooLargeForCellLimit,
    retry,
    fieldContext,
    isStale,
    isUpdating,
  };
}

export default useOceanMapLayer;
