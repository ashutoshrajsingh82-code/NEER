"use client";

// -----------------------------------------------------------------------------
// NEER — DateDepthContext  (Phase 36A)
//
// The one shared instance of the backend-driven date/depth state model
// (lib/useDateDepth.js over lib/dateDepthModel.js). Mounted once in the (app)
// route-group layout, ABOVE PointInspectionProvider, so every consumer —
// OceanMap, PointInspection, the KPI drawer, and the scientific pages later
// phases add (Hovmöller, vertical profile, 3D ocean, ...) — reads the same
// availableDates / selectedDate / availableDepths / selectedDepth, and the
// backend is asked for /dates and /model/info exactly once.
//
// Deliberately separate from PointInspectionContext: date and depth are
// app-wide "what slice of the ocean are we looking at" state; a selected map
// point is dashboard-specific. PointInspectionContext consumes this one and
// re-exposes `date`/`depth` under their original names for its existing
// consumers.
// -----------------------------------------------------------------------------

import { createContext, useContext } from "react";
import { useDateDepth } from "@/lib/useDateDepth";

const DateDepthContext = createContext(null);

export function DateDepthProvider({ children }) {
  const value = useDateDepth();
  return <DateDepthContext.Provider value={value}>{children}</DateDepthContext.Provider>;
}

/**
 * @returns {ReturnType<typeof useDateDepth>} — see lib/useDateDepth.js for
 *   the full field list. Key fields:
 *   - `selectedDate`: "YYYY-MM-DD" | null   (null until /dates succeeds)
 *   - `selectedDepth`: number (m) | null
 *   - `availableDates`: string[] (ascending), `availableDepths`: number[]
 *   - `isReady`: both selected — safe to run date/depth-parameterised queries
 *   - `selectDate(date)` / `selectDepth(depth)`: only values in the
 *     available lists are accepted; return `true` if honoured
 *   - `stepDate(±1)`: move to the adjacent *available* date
 *   - `datesStatus` / `depthsStatus`: "loading" | "success" | "empty" | "error"
 */
export function useDateDepthContext() {
  const context = useContext(DateDepthContext);
  if (!context) {
    throw new Error("useDateDepthContext must be used within a <DateDepthProvider>.");
  }
  return context;
}