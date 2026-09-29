"use client";

import { useVisualizationVariable } from "@/app/(app)/dashboard/_context/VisualizationVariableContext";

export default function VisualizationModeControl() {
  const { variableMode, setVariableMode } = useVisualizationVariable();
  return <fieldset className="flex flex-wrap items-center gap-2 text-caption text-text-secondary">
    <legend className="sr-only">Visualization variable</legend>
    <span>Display:</span>
    {["temperature", "anomaly"].map((mode) => <button key={mode} type="button" aria-pressed={variableMode === mode} onClick={() => setVariableMode(mode)} className={`rounded border px-3 py-1.5 ${variableMode === mode ? "border-accent-400 bg-accent-900/50 text-text-primary" : "border-border-subtle text-text-secondary"}`}>
      {mode === "temperature" ? "MODEL OUTPUT" : "ANOMALY"}
    </button>)}
  </fieldset>;
}
