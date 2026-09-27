"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — PointInspectionContext  (Phase 34D foundation, Phase 34E
// completes the synchronization chain)
//
// The one piece of cross-tree state the dashboard route needs: OceanMap
// (rendered inside app/(app)/dashboard/page.js, a Server Component, via the
// OceanMapSection client wrapper), DataContextPanel (also page.js, via the
// DataContextSection wrapper), PointInspection, and the KPI drawer (both
// rendered inside the shared shell's slots, in app/(app)/layout.js) are
// siblings under AppShell, not parent/child — a date change or a map click
// has to reach components none of them render directly. A context provided
// above all of them is the standard fix (see OceanMap.js's Phase 34C header
// comment: "wiring [selection] to the shared InspectionPanel is left to
// whichever later phase owns that integration" — Phase 34D was that phase
// for selection; Phase 34E is this file's counterpart for date/depth).
//
// Phase 34E additions over Phase 34D:
//   - `date`/`depth` are now real state (`setDate`/`setDepth`), not just
//     Provider props nothing ever changed after mount. DataContextSection
//     wires DataContextPanel's prev/next-day and depth-level controls to
//     these, completing "Date selection -> Map context -> Point selection ->
//     Inspection panel -> KPI information" (the phase's own diagram): a date
//     change here is what OceanMapSection's `date` prop, PointInspection's
//     `date` prop, and the KPI drawer's "Selected Date" card all read.
//   - `pointInspection` holds the *shared* `reconstruct()` (frontend/lib/
//     api.js, Phase 33B) request state for the current selection, fetched
//     here via `useApiRequest` and re-run whenever the selected point, date,
//     or depth changes. Both PointInspection (its temperature/anomaly cards)
//     and the KPI drawer (its "Reconstructed Temp."/"Anomaly" KPIs) need the
//     same result for the same selection — fetching it once here, instead of
//     once per consumer, is this phase's fix for the duplicate-fetch problem
//     the two consumers would otherwise have (see PointInspection.js's own
//     "Requirement 10" comment, carried over from Phase 34D). PointInspection
//     itself still has no *import* of this context (see below) — the shared
//     fetch state reaches it as a plain prop from layout.js, same as
//     selectedPoint/date/depth already did.
//
// Deliberately still narrow, per useApiRequest.js's Requirement 1 precedent
// ("do not create unnecessary global state"): selection, the current
// date/depth/data-mode "view", and the one per-selection fetch every
// dashboard panel needs — nothing else. `metrics()` (RMSE/MAE) is NOT held
// here: it doesn't depend on the current selection, so DashboardKPIs.js
// fetches it locally instead of growing this context with state most of its
// other consumers have no use for.
//
// Scoped to the dashboard route on purpose (lives under dashboard/_context,
// not components/shell) — a future Reconstruction/Explainability/Evaluation
// page that wants the same map<->inspector<->KPI wiring stands up its own
// provider the same way rather than sharing this one, since each of those
// workflows will have its own notion of "what's selected".
// -----------------------------------------------------------------------------

import { createContext, useCallback, useContext, useMemo, useState } from "react";
import { reconstruct } from "@/lib/api";
import { useApiRequest } from "@/lib/useApiRequest";

const PointInspectionContext = createContext(null);

// Same placeholder "current view" values page.js (Phase 34C) and
// DataContextPanel (Phase 34B) each hardcoded independently before this
// context centralized them (Phase 34D for the constants, Phase 34E for
// actually being able to change them).
const DEFAULT_DATA_MODE = "reconstructed";
const DEFAULT_DATE = "2024-03-18";
const DEFAULT_DEPTH = 0;

/**
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 * @param {string|Date} date - initial selected date
 * @param {number} depth - initial selected depth, metres
 */
export function PointInspectionProvider({
  children,
  dataMode = DEFAULT_DATA_MODE,
  date: initialDate = DEFAULT_DATE,
  depth: initialDepth = DEFAULT_DEPTH,
}) {
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [date, setDateState] = useState(initialDate);
  const [depth, setDepthState] = useState(initialDepth);

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

  // The one place `reconstruct()` is kicked off from. Called synchronously
  // from selectPoint/setDate/setDepth below (never from an effect) so the
  // status flip to "loading" lands in the *same* render as the selection/
  // date/depth change that triggered it — an async function's body runs
  // synchronously up to its first `await`, so `run`'s `setStatus(LOADING)`
  // batches with whatever state update called us here. That avoids the
  // one-frame-of-stale-data gap Phase 34D's PointInspection.js used a `key`-
  // forced remount to work around; consumers here just need to keep
  // checking isLoading/isIdle before isSuccess, same as before.
  const fetchPoint = useCallback(
    (point, atDate, atDepth) => {
      runReconstruct({ lat: point.lat, lon: point.lon, date: atDate, depth: atDepth }).catch(() => {
        // Swallowed here on purpose — pointError/pointIsError above already
        // capture the failure for consumers (PointInspection, the KPI
        // drawer) to render their own error states.
      });
    },
    [runReconstruct]
  );

  const selectPoint = useCallback(
    (point) => {
      setSelectedPoint(point);
      if (point) {
        fetchPoint(point, date, depth);
      } else {
        resetReconstruct();
      }
    },
    [date, depth, fetchPoint, resetReconstruct]
  );

  const clearSelection = useCallback(() => {
    setSelectedPoint(null);
    resetReconstruct();
  }, [resetReconstruct]);

  // Changing the date/depth while a point is already selected re-inspects
  // the same point at the new date/depth — this is the "Date selection ->
  // ... -> KPI information" chain the phase spec describes: nobody has to
  // re-click the map for the inspection panel/KPIs to catch up.
  const setDate = useCallback(
    (nextDate) => {
      setDateState(nextDate);
      if (selectedPoint) fetchPoint(selectedPoint, nextDate, depth);
    },
    [selectedPoint, depth, fetchPoint]
  );

  const setDepth = useCallback(
    (nextDepth) => {
      setDepthState(nextDepth);
      if (selectedPoint) fetchPoint(selectedPoint, date, nextDepth);
    },
    [selectedPoint, date, fetchPoint]
  );

  const pointInspection = useMemo(
    () => ({
      data: pointData,
      error: pointError,
      isLoading: pointLoading,
      isIdle: pointIdle,
      isSuccess: pointSuccess,
      isError: pointIsError,
      /** Re-runs the same request — for a "Retry" button after a failure. */
      retry: () => {
        if (selectedPoint) fetchPoint(selectedPoint, date, depth);
      },
    }),
    [pointData, pointError, pointLoading, pointIdle, pointSuccess, pointIsError, selectedPoint, date, depth, fetchPoint]
  );

  const value = useMemo(
    () => ({
      selectedPoint,
      selectPoint,
      clearSelection,
      dataMode,
      date,
      setDate,
      depth,
      setDepth,
      pointInspection,
    }),
    [selectedPoint, selectPoint, clearSelection, dataMode, date, setDate, depth, setDepth, pointInspection]
  );

  return <PointInspectionContext.Provider value={value}>{children}</PointInspectionContext.Provider>;
}

/**
 * @returns {{
 *   selectedPoint: {lat: number, lon: number}|null,
 *   selectPoint: (point: {lat: number, lon: number}) => void,
 *   clearSelection: () => void,
 *   dataMode: "reconstructed"|"observed"|"blended",
 *   date: string|Date,
 *   setDate: (date: string|Date) => void,
 *   depth: number,
 *   setDepth: (depth: number) => void,
 *   pointInspection: {
 *     data: object|null,
 *     error: import("@/lib/api").ApiError|null,
 *     isLoading: boolean,
 *     isIdle: boolean,
 *     isSuccess: boolean,
 *     isError: boolean,
 *     retry: () => void,
 *   },
 * }}
 */
export function usePointInspection() {
  const context = useContext(PointInspectionContext);
  if (!context) {
    throw new Error("usePointInspection must be used within a <PointInspectionProvider>.");
  }
  return context;
}