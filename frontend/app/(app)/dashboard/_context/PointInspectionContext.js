"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — PointInspectionContext  (Phase 34D foundation, Phase 34E
// synchronization chain, Phase 36A backend-driven date/depth)
//
// Cross-tree state for the dashboard route: the selected map point, the
// current data-mode "view", and the one shared `reconstruct()` request for
// the current selection. OceanMap (via OceanMapSection), DataContextSection
// (via DataContextSection), PointInspection and the KPI drawer are siblings
// under AppShell, so a context above all of them is how a change reaches
// components none of them render directly (see app/(app)/layout.js).
//
// Phase 36A: date and depth are NO LONGER owned here. They live in
// DateDepthContext (app/(app)/_context/DateDepthContext.js) — backend-driven
// (GET /dates, GET /model/info), validated, and shared app-wide. This
// provider *consumes* that and re-exposes `date`/`depth`/`setDate`/`setDepth`
// under their original names so OceanMapSection, DashboardKPIs and layout.js
// keep working unchanged. What changed underneath:
//   - `date`/`depth` are `null` until the backend has supplied valid values —
//     there is no hard-coded default date any more. `date` is always the
//     canonical "YYYY-MM-DD" string.
//   - the old ad-hoc `dates()` effect (which re-fired on every date change)
//     is gone; DateDepthProvider fetches once.
//   - `reconstruct()` is now driven by ONE effect keyed on the whole
//     selection (point + date + depth), instead of being kicked off by hand
//     from selectPoint/setDate/setDepth. That's what lets it also fire when
//     the date/depth *arrive* from the backend after a point was already
//     clicked, or get reconciled (e.g. selected depth vanishes from a new
//     depth list) — cases the hand-wired version couldn't see.
//     The one-frame stale-data gap that the synchronous approach avoided is
//     handled by `isPending` below: any render whose selection differs from
//     the one last requested reports "loading" and hides the previous
//     result, so a consumer can never show point A's number under point B.
// -----------------------------------------------------------------------------

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, demoContext as fetchDemoContext, runReconstruction } from "@/lib/api";
import { snapToGrid } from "@/lib/oceanDomain";
import { useApiRequest } from "@/lib/useApiRequest";
import { classifyPointLocation, isQueryablePoint } from "@/lib/pointClassification";
import { isOnLand } from "@/components/ocean-map/landmask";
import { useDateDepthContext } from "./DateDepthContext";

const PointInspectionContext = createContext(null);

const DEFAULT_DATA_MODE = "reconstructed";

/** Settle time for a changed point/date/depth before GET /reconstruct is re-issued (see the effect below). */
const POINT_REQUEST_DEBOUNCE_MS = 150;

function reconstructInspectionPoint({ lat, lon, date, depth }) {
  return runReconstruction({ date, depth, latitude: lat, longitude: lon });
}

/**
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 */
export function PointInspectionProvider({ children, dataMode = DEFAULT_DATA_MODE }) {
  const {
    selectedDate: date,
    selectedDepth: depth,
    selectDate,
    selectDepth,
    datesError,
    retryDates,
    modelInfoData,
    availableDates,
    availableDepths,
    datesStatus,
    depthsStatus,
  } = useDateDepthContext();

  const [selectedPoint, setSelectedPoint] = useState(null);
  const [variableMode, setVariableModeState] = useState("temperature");
  const [demoMode, setDemoMode] = useState(false);
  const [demoStatus, setDemoStatus] = useState({ state: "idle", error: null, context: null });
  const defaultPointInitialized = useRef(false);
  const setVariableMode = useCallback((mode) => {
    if (mode === "temperature" || mode === "anomaly") setVariableModeState(mode);
  }, []);
  const activateDemoMode = useCallback(async () => {
    setDemoStatus({ state: "loading", error: null, context: null });
    setDemoMode(false);
    try {
      const context = await fetchDemoContext();
      if (context?.available !== true || context?.variable_mode !== "temperature" || context?.depth !== 100) {
        throw new Error("The backend returned an incomplete demonstration context.");
      }
      if (!Number.isFinite(context.latitude) || !Number.isFinite(context.longitude)) {
        throw new Error("The backend demonstration location is invalid.");
      }
      if (datesStatus !== "success" || !availableDates.includes(context.date) || depthsStatus !== "success" || !availableDepths.includes(context.depth)) {
        throw new Error("The demonstration date or 100 m depth is not available in the shared backend metadata.");
      }
      const locationStatus = classifyPointLocation(context.latitude, context.longitude, isOnLand);
      if (!isQueryablePoint(locationStatus)) throw new Error("The backend demonstration location is not a valid ocean point.");
      selectDate(context.date);
      selectDepth(context.depth);
      setSelectedPoint({ lat: context.latitude, lon: context.longitude });
      setVariableModeState("temperature");
      setDemoMode(true);
      setDemoStatus({ state: "ready", error: null, context });
      return true;
    } catch (error) {
      setDemoStatus({ state: "error", error: error?.message || "Demo mode unavailable.", context: null });
      return false;
    }
  }, [selectDate, selectDepth, availableDates, availableDepths, datesStatus, depthsStatus]);

  const {
    data: pointData,
    error: pointError,
    isLoading: pointLoading,
    isIdle: pointIdle,
    isSuccess: pointSuccess,
    isError: pointIsError,
    run: runReconstruct,
    reset: resetReconstruct,
  } = useApiRequest(reconstructInspectionPoint);
  const {
    data: surfaceData,
    error: surfaceError,
    isLoading: surfaceLoading,
    run: runSurface,
    reset: resetSurface,
  } = useApiRequest(reconstructInspectionPoint);

  // Single source of truth for "is this selection ocean, land, or outside
  // the NEER domain" (lib/pointClassification.js). `null` when nothing is
  // selected.
  const pointStatus = useMemo(
    () => (selectedPoint ? classifyPointLocation(selectedPoint.lat, selectedPoint.lon, isOnLand) : null),
    [selectedPoint]
  );

  const selectPoint = useCallback((point) => setSelectedPoint(point ?? null), []);
  const clearSelection = useCallback(() => setSelectedPoint(null), []);

  // Seed a usable first view for routes opened directly, without enabling
  // SIH demo mode. The demo endpoint provides its ocean grid cell when
  // available; a known in-domain ocean point is the fallback.
  useEffect(() => {
    if (selectedPoint || defaultPointInitialized.current || datesStatus !== "success" || depthsStatus !== "success" || !date || depth == null) return undefined;
    let cancelled = false;
    const chooseDefault = (context) => {
      if (cancelled || defaultPointInitialized.current || selectedPoint) return;
      const candidates = [
        ...(Number.isFinite(context?.latitude) && Number.isFinite(context?.longitude)
          ? [{ lat: context.latitude, lon: context.longitude }]
          : []),
        { lat: 15.5, lon: 72.75 },
      ];
      const point = candidates
        .map(({ lat, lon }) => snapToGrid(lat, lon))
        .find(({ lat, lon }) => isQueryablePoint(classifyPointLocation(lat, lon, isOnLand)));
      defaultPointInitialized.current = true;
      if (point) setSelectedPoint(point);
    };
    fetchDemoContext().then(chooseDefault).catch(() => chooseDefault(null));
    return () => { cancelled = true; };
  }, [selectedPoint, datesStatus, depthsStatus, date, depth]);

  // The complete inputs of one reconstruct() call, or null when there's
  // nothing valid to ask for: no point, a land/outside-domain point (GET
  // /reconstruct has no notion of land — asking would return a number that
  // isn't a real ocean measurement), or a date/depth the backend hasn't
  // supplied yet.
  const requestKey =
    selectedPoint && isQueryablePoint(pointStatus) && date !== null && depth !== null
      ? `${selectedPoint.lat}|${selectedPoint.lon}|${date}|${depth}`
      : null;

  const [requestedKey, setRequestedKey] = useState(null);
  const [retryToken, setRetryToken] = useState(0);

  // Phase 36B: dragging the depth slider or holding an arrow key on the depth
  // chips changes the key many times a second. Once a first request has been
  // made, a change waits briefly so only the value the user settles on is
  // requested; `isPending` (requestedKey lagging requestKey) reports "loading"
  // for that whole gap, so nothing stale is shown while waiting. The very
  // first request for a selection goes out immediately.
  const hasRequestedRef = useRef(false);
  useEffect(() => {
    if (requestKey === null) {
      resetReconstruct();
      resetSurface();
      setRequestedKey(null);
      hasRequestedRef.current = false;
      return undefined;
    }
    const timer = setTimeout(
      () => {
        hasRequestedRef.current = true;
        setRequestedKey(requestKey);
        runReconstruct({ lat: selectedPoint.lat, lon: selectedPoint.lon, date, depth }).catch(() => {
          // Swallowed on purpose — pointError/pointIsError capture the failure
          // for consumers to render their own error states.
        });
        if (depth !== 0 && !(surfaceData?.date === date && surfaceData?.lat === selectedPoint.lat && surfaceData?.lon === selectedPoint.lon)) {
          runSurface({ lat: selectedPoint.lat, lon: selectedPoint.lon, date, depth: 0 }).catch(() => {});
        }
      },
      hasRequestedRef.current ? POINT_REQUEST_DEBOUNCE_MS : 0
    );
    return () => clearTimeout(timer);
    // requestKey encodes lat/lon/date/depth; the rest are stable callbacks.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey, retryToken]);

  const isPending = requestKey !== null && requestedKey !== requestKey;

  // A queryable point is selected but there's no valid date/depth to ask
  // with. Consumers treat "idle" as "loading", so map this explicitly:
  // still waiting on /dates -> loading; /dates failed or is empty -> an
  // error the panel can show (with a working Retry) instead of an
  // endless skeleton.
  const awaitingSelection =
    selectedPoint !== null && isQueryablePoint(pointStatus) && (date === null || depth === null);
  const selectionUnavailable = awaitingSelection && (datesStatus === "error" || datesStatus === "empty");
  const selectionError = useMemo(() => {
    if (!selectionUnavailable) return null;
    return datesStatus === "empty"
      ? new ApiError({ code: "no_dates_available", message: "The backend reports no available dates, so this point can't be inspected." })
      : new ApiError({
          code: datesError?.code ?? "dates_unavailable",
          message: `Available dates couldn't be loaded (${datesError?.message ?? "unknown error"}), so this point can't be inspected.`,
          status: datesError?.status ?? null,
        });
  }, [selectionUnavailable, datesStatus, datesError]);

  const pointInspection = useMemo(() => {
    if (awaitingSelection) {
      return {
        data: null,
        error: selectionError,
        isLoading: !selectionUnavailable,
        isIdle: false,
        isSuccess: false,
        isError: selectionUnavailable,
        retry: retryDates,
      };
    }
    return {
      // `data` is non-null ONLY on success for the *current* selection —
      // useApiRequest keeps the previous result around while a refetch is
      // loading, and a consumer reading `data` directly must never see
      // another point's/date's/depth's value.
      data: pointSuccess && !isPending ? pointData : null,
      surfaceTemperature:
        !isPending && depth === 0 && pointSuccess
          ? pointData?.temperature
          : !isPending && depth !== 0 && surfaceData?.date === date && surfaceData?.lat === selectedPoint?.lat && surfaceData?.lon === selectedPoint?.lon
            ? surfaceData.temperature
            : null,
      surfaceIsLoading: depth !== 0 && (surfaceLoading || isPending),
      surfaceError: depth !== 0 && !isPending ? surfaceError : null,
      error: pointIsError && !isPending ? pointError : null,
      isLoading: pointLoading || isPending,
      isIdle: pointIdle && !isPending,
      isSuccess: pointSuccess && !isPending,
      isError: pointIsError && !isPending,
      /** Re-runs the same request — for a "Retry" button after a failure. */
      retry: () => setRetryToken((n) => n + 1),
    };
  }, [
    awaitingSelection,
    date,
    depth,
    selectedPoint?.lat,
    selectedPoint?.lon,
    selectionError,
    selectionUnavailable,
    retryDates,
    isPending,
    pointData,
    surfaceData,
    surfaceLoading,
    surfaceError,
    pointError,
    pointLoading,
    pointIdle,
    pointSuccess,
    pointIsError,
  ]);

  const value = useMemo(
    () => ({
      selectedPoint,
      selectPoint,
      clearSelection,
      dataMode,
      date,
      setDate: selectDate,
      depth,
      setDepth: selectDepth,
      pointInspection,
      pointStatus,
      modelInfoData,
      variableMode,
      setVariableMode,
      demoMode,
      demoStatus,
      activateDemoMode,
      deactivateDemoMode: () => {
        setDemoMode(false);
        setDemoStatus({ state: "idle", error: null, context: null });
      },
    }),
    [selectedPoint, selectPoint, clearSelection, dataMode, date, selectDate, depth, selectDepth, pointInspection, pointStatus, modelInfoData, variableMode, setVariableMode, demoMode, demoStatus, activateDemoMode]
  );

  return <PointInspectionContext.Provider value={value}>{children}</PointInspectionContext.Provider>;
}

/**
 * @returns {{
 *   selectedPoint: {lat: number, lon: number}|null,
 *   selectPoint: (point: {lat: number, lon: number}|null) => void,
 *   clearSelection: () => void,
 *   dataMode: "reconstructed"|"observed"|"blended",
 *   date: string|null,            // "YYYY-MM-DD" from DateDepthContext; null until /dates succeeds
 *   setDate: (date: string|Date) => boolean,   // only available dates accepted
 *   depth: number|null,           // metres, from DateDepthContext
 *   setDepth: (depth: number) => boolean,      // only available depths accepted
 *   pointInspection: {
 *     data: object|null,
 *     error: import("@/lib/api").ApiError|null,
 *     isLoading: boolean,
 *     isIdle: boolean,
 *     isSuccess: boolean,
 *     isError: boolean,
 *     retry: () => void,
 *   },
 *   pointStatus: "ocean"|"land"|"outside_domain"|null,
 *   variableMode: "temperature"|"anomaly",
 *   setVariableMode: (mode: "temperature"|"anomaly") => void,
 *   demoMode: boolean,
 *   demoStatus: {state: "idle"|"loading"|"ready"|"error", error: string|null},
 *   activateDemoMode: () => Promise<boolean>,
 * }}
 */
export function usePointInspection() {
  const context = useContext(PointInspectionContext);
  if (!context) {
    throw new Error("usePointInspection must be used within a <PointInspectionProvider>.");
  }
  return context;
}
