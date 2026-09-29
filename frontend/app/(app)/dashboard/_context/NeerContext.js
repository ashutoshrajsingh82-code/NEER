"use client";

import { useDateDepthContext } from "./DateDepthContext";
import { usePointInspection } from "./PointInspectionContext";

/** One application-facing view of the persisted NEER scientific selection. */
export function useNeerContext() {
  const dates = useDateDepthContext();
  const selection = usePointInspection();
  return {
    date: selection.date,
    latitude: selection.selectedPoint?.lat ?? null,
    longitude: selection.selectedPoint?.lon ?? null,
    depth: selection.depth,
    variableMode: selection.variableMode,
    setDate: selection.setDate,
    setLocation: selection.selectPoint,
    setDepth: selection.setDepth,
    setVariableMode: selection.setVariableMode,
    selectedPoint: selection.selectedPoint,
    selectPoint: selection.selectPoint,
    clearSelection: selection.clearSelection,
    pointInspection: selection.pointInspection,
    pointStatus: selection.pointStatus,
    dataMode: selection.dataMode,
    modelInfoData: selection.modelInfoData,
    demoMode: selection.demoMode,
    demoStatus: selection.demoStatus,
    activateDemoMode: selection.activateDemoMode,
    deactivateDemoMode: selection.deactivateDemoMode,
    ...dates,
  };
}
