"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — PointInspectionContext  (Phase 34D)
//
// The one piece of cross-tree state Phase 34D needs: OceanMap (rendered
// inside app/(app)/dashboard/page.js, a Server Component, via the
// OceanMapSection client wrapper) and PointInspection (rendered inside the
// shared shell's InspectionPanel slot, in app/(app)/layout.js) are siblings
// under AppShell, not parent/child — a click on the map has to reach a
// component the map itself never renders. A context provided above both is
// the standard fix (see OceanMap.js's Phase 34C header comment: "wiring
// [selection] to the shared InspectionPanel is left to whichever later
// phase owns that integration" — this is that phase).
//
// Deliberately narrow, per useApiRequest.js's Requirement 1 precedent ("do
// not create unnecessary global state"): this holds selection plus the
// current date/depth/data-mode "view" (so OceanMap and PointInspection
// always agree on what they're both showing/inspecting), and nothing else.
// The date/depth/mode values are still the same Phase 34B/C placeholders
// DataContextPanel and OceanMap each defaulted to on their own before this
// phase — centralizing them here just means the map and the inspector can
// no longer silently disagree. Wiring them to DataContextPanel's own
// controls (so changing the date there updates what's inspected) is left to
// a later phase, same as DataContextPanel's own header comment already
// scopes it.
//
// Scoped to the dashboard route on purpose (lives under dashboard/_context,
// not components/shell) — a future Reconstruction/Explainability/Evaluation
// page that wants the same map<->inspector wiring stands up its own
// provider the same way rather than sharing this one, since each of those
// workflows will have its own notion of "what's selected". PointInspection
// itself (components/inspection) has no dependency on this context — it's
// plain props in, so any future page can drive it from whatever state shape
// fits that workflow.
// -----------------------------------------------------------------------------

import { createContext, useCallback, useContext, useMemo, useState } from "react";

const PointInspectionContext = createContext(null);

// Same placeholder "current view" values page.js (Phase 34C) and
// DataContextPanel (Phase 34B) each hardcoded independently.
const DEFAULT_DATA_MODE = "reconstructed";
const DEFAULT_DATE = "2024-03-18";
const DEFAULT_DEPTH = 0;

/**
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 * @param {string|Date} date
 * @param {number} depth - metres
 */
export function PointInspectionProvider({
  children,
  dataMode = DEFAULT_DATA_MODE,
  date = DEFAULT_DATE,
  depth = DEFAULT_DEPTH,
}) {
  const [selectedPoint, setSelectedPoint] = useState(null);

  const selectPoint = useCallback((point) => setSelectedPoint(point), []);
  const clearSelection = useCallback(() => setSelectedPoint(null), []);

  const value = useMemo(
    () => ({ selectedPoint, selectPoint, clearSelection, dataMode, date, depth }),
    [selectedPoint, selectPoint, clearSelection, dataMode, date, depth]
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
 *   depth: number,
 * }}
 */
export function usePointInspection() {
  const context = useContext(PointInspectionContext);
  if (!context) {
    throw new Error("usePointInspection must be used within a <PointInspectionProvider>.");
  }
  return context;
}