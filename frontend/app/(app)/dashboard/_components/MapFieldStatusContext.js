"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — MapFieldStatusContext  (Phase 36B)
//
// Carries ONE fact from the OceanMap to the date/depth controls: whether the
// map is currently fetching (or still showing a superseded field while it
// fetches) the scientific field for the selected date/depth, and whether that
// fetch failed. The map owns the request (useOceanMapLayer -> lib/api.js
// reconstructGrid); the controls only need to show a small "updating" cue next
// to what the user just changed. Deliberately read-only for the controls — they
// never trigger or configure the request, which follows from date/depth alone.
// -----------------------------------------------------------------------------

import { createContext, useContext, useMemo, useState } from "react";

const MapFieldStatusContext = createContext(null);
const IDLE = Object.freeze({ isUpdating: false, isError: false });

export function MapFieldStatusProvider({ children }) {
  const [status, setStatus] = useState(IDLE);
  const value = useMemo(() => ({ ...status, setStatus }), [status]);
  return <MapFieldStatusContext.Provider value={value}>{children}</MapFieldStatusContext.Provider>;
}

/** Outside a provider (e.g. OceanMap used standalone) this is a harmless idle no-op. */
export function useMapFieldStatus() {
  return useContext(MapFieldStatusContext) ?? { ...IDLE, setStatus: () => {} };
}