"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — OceanMapSection  (Phase 34D)
//
// Connects OceanMap to PointInspectionContext. Exists only because of a
// Server/Client boundary constraint: dashboard/page.js is a Server Component
// (it exports `metadata`, which Next.js only allows there — see this
// project's Phase 34D/QA history of framer-motion "use client" bugs for why
// that boundary is worth respecting rather than routing around), so it
// cannot itself call the usePointInspection() hook. OceanMap already has to
// be a Client Component (it uses framer-motion-free but stateful pan/zoom/
// hover interaction) and page.js already renders it directly — this
// component just sits between them so the hook usage lives in a Client
// Component, not in page.js.
//
// OceanMap's own controlled/uncontrolled `selectedPoint`/`onSelectPoint`
// contract (Phase 34C) is unchanged and still works standalone without this
// wrapper or the context — this file is purely dashboard-route wiring.
// -----------------------------------------------------------------------------

import { OceanMap } from "@/components/ocean-map";
import { useNeerContext } from "../_context/NeerContext";
import { useMapFieldStatus } from "./MapFieldStatusContext";
import { useShell } from "@/components/shell/ShellContext";
import { useEffect } from "react";

export default function OceanMapSection() {
  const { dataMode, date, depth, selectedPoint, selectPoint, variableMode, setVariableMode } = useNeerContext();
  const { setInspectionOpen } = useShell();
  const { setStatus } = useMapFieldStatus();

  useEffect(() => {
    if (selectedPoint) setInspectionOpen(true);
  }, [selectedPoint, setInspectionOpen]);

  function handleSelectPoint(point) {
    selectPoint(point);
    setInspectionOpen(true);
  }

  return (
    <OceanMap
      dataMode={dataMode}
      variable={variableMode === "anomaly" ? "anomaly" : "sst"}
      onVariableChange={(value) => {
        if (value === "anomaly") setVariableMode("anomaly");
        if (value === "sst") setVariableMode("temperature");
      }}
      date={date}
      depth={depth}
      selectedPoint={selectedPoint}
      onSelectPoint={handleSelectPoint}
      onFieldStatusChange={setStatus}
    />
  );
}
