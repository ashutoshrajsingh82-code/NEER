"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — DataContextSection  (Phase 34E)
//
// Connects DataContextPanel to PointInspectionContext — the same Server/
// Client boundary wrapper pattern OceanMapSection.js established in Phase
// 34D: dashboard/page.js is a Server Component (it exports `metadata`), so
// it can't call usePointInspection() itself.
//
// Before this file, DataContextPanel's date/depth controls were Phase 34B
// local state that never left the component (page.js rendered it
// uncontrolled). Making it controlled here is what completes the phase's
// "Date selection -> Map context -> Point selection -> Inspection panel ->
// KPI information" chain: a date/depth change here now flows into
// OceanMapSection's map, PointInspection's fetch, and the KPI drawer's
// "Selected Date"/"Selected Depth" cards, all through the one shared
// context.
// -----------------------------------------------------------------------------

import DataContextPanel from "./DataContextPanel";
import { usePointInspection } from "../_context/PointInspectionContext";

export default function DataContextSection({ className }) {
  const { dataMode, date, setDate, depth, setDepth } = usePointInspection();

  return (
    <DataContextPanel
      dataMode={dataMode}
      date={date}
      onDateChange={setDate}
      depth={depth}
      onDepthChange={setDepth}
      className={className}
    />
  );
}