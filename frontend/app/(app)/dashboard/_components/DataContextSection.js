"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — DataContextSection  (Phase 34E, backend-driven in 36A)
//
// Connects DataContextPanel to the shared state. Same Server/Client boundary
// wrapper pattern OceanMapSection.js established in Phase 34D: dashboard/
// page.js is a Server Component (it exports `metadata`), so it can't call a
// context hook itself.
//
// Phase 36A: date/depth now come from DateDepthContext (GET /dates and GET
// /model/info, validated — see lib/dateDepthModel.js) rather than local
// state, so the controls can only ever land on a date the backend lists and
// a depth the model supports. `dataMode` still comes from
// PointInspectionContext (it's a dashboard "view" setting, not backend-driven).
// A change here reaches OceanMapSection, PointInspection and the KPI drawer
// through the same shared context.
// -----------------------------------------------------------------------------

import DataContextPanel from "./DataContextPanel";
import { usePointInspection } from "../_context/PointInspectionContext";
import { useDateDepthContext } from "../../_context/DateDepthContext";

const DATES_MESSAGES = {
  network: "Can't reach the NEER backend.",
  timeout: "The backend took too long to respond.",
  config: "Backend URL isn't configured.",
  invalid_response: "Backend sent an invalid date list.",
  api: "Backend couldn't provide dates.",
};

export default function DataContextSection({ className }) {
  const { dataMode } = usePointInspection();
  const {
    selectedDate,
    stepDate,
    canStepPrev,
    canStepNext,
    datesStatus,
    datesError,
    datesErrorCategory,
    retryDates,
    depthOptions,
    selectedDepth,
    selectDepth,
    depthsStatus,
    depthsSource,
    depthsValidation,
  } = useDateDepthContext();

  const datesMessage =
    datesStatus === "empty"
      ? "Backend reports no available dates."
      : datesStatus === "error"
        ? datesError?.message || DATES_MESSAGES[datesErrorCategory] || "Dates unavailable."
        : undefined;

  // Never present a fallback or mismatching list as if it were the backend's
  // confirmed configuration.
  const depthsMessage =
    depthsSource === "expected_fallback"
      ? "Using expected NEER depth levels — model info unavailable."
      : depthsValidation.matchesExpected === false
        ? "Backend depth levels differ from the expected NEER configuration."
        : undefined;

  return (
    <DataContextPanel
      dataMode={dataMode}
      date={selectedDate}
      onStepDate={stepDate}
      canStepPrev={canStepPrev}
      canStepNext={canStepNext}
      datesStatus={datesStatus}
      datesMessage={datesMessage}
      onRetryDates={retryDates}
      depthLevels={depthOptions}
      depth={selectedDepth ?? undefined}
      onDepthChange={selectDepth}
      depthsStatus={depthsStatus}
      depthsMessage={depthsMessage}
      className={className}
    />
  );
}