"use client";

// Backwards-compatible hook name; variable mode is stored in the single
// application-wide PointInspectionContext alongside location/date/depth.
import { usePointInspection } from "./PointInspectionContext";

export function useVisualizationVariable() {
  const { variableMode, setVariableMode } = usePointInspection();
  return { variableMode, setVariableMode };
}
