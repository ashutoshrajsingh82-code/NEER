"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — PointInspectionContext  (Phase 34D foundation, Phase 34E
// synchronization chain, Phase 36A backend-driven date/depth)
//
// Cross-tree state for the dashboard route: the selected map point, the
// current data-mode "view", and the one shared `reconstruct()` request for
// the current selection. OceanMap (via OceanMapSection), DataContextPanel
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

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { ApiError, reconstruct } from "@/lib/api";
import { useApiRequest } from "@/lib/useApiRequest";
import { classifyPointLocation, isQueryablePoint } from "@/lib/pointClassification";
import { isOnLand } from "@/components/ocean-map/landmask";
import { useDateDepthContext } from "../../_context/DateDepthContext";

const PointInspectionContext = createContext(null);

const DEFAULT_DATA_MODE = "reconstructed";

/**
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 */
export function PointInspectionProvider({ children, dataMode = DEFAULT_DATA_MODE }) {
  const {
    selectedDate: date,
    selectedDepth: depth,
    selectDate,
    selectDepth,
    datesStatus,
    datesError,
    retryDates,
  } = useDateDepthContext();

  const [selectedPoint, setSelectedPoint] = useState(null);

  const {
    data: pointData,
    error: pointError,
    isLoading: pointLoading,
    isIdle: pointIdle,
    isSuccess: pointSuccess,
    isError: pointIsError,
    run: runReconstruct,
    reset: resetReconstruct,
  } = useApiRequest(reconstruct);

  // Single source of truth for "is this selection ocean, land, or outside
  // the NEER domain" (lib/pointClassification.js). `null` when nothing is
  // selected.
  const pointStatus = useMemo(
    () => (selectedPoint ? classifyPointLocation(selectedPoint.lat, selectedPoint.lon, isOnLand) : null),
    [selectedPoint]
  );

  const selectPoint = useCallback((point) => setSelectedPoint(point ?? null), []);
  const clearSelection = useCallback(() => setSelectedPoint(null), []);

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

  useEffect(() => {
    if (requestKey === null) {
      resetReconstruct();
      setRequestedKey(null);
      return;
    }
    setRequestedKey(requestKey);
    runReconstruct({ lat: selectedPoint.lat, lon: selectedPoint.lon, date, depth }).catch(() => {
      // Swallowed on purpose — pointError/pointIsError capture the failure
      // for consumers to render their own error states.
    });
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
    selectionError,
    selectionUnavailable,
    retryDates,
    isPending,
    pointData,
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
    }),
    [selectedPoint, selectPoint, clearSelection, dataMode, date, selectDate, depth, selectDepth, pointInspection, pointStatus]
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
 * }}
 */
export function usePointInspection() {
  const context = useContext(PointInspectionContext);
  if (!context) {
    throw new Error("usePointInspection must be used within a <PointInspectionProvider>.");
  }
  return context;
}