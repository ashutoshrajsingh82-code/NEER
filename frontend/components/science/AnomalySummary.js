"use client";

import { useMemo } from "react";
import { summarizeFiniteValues } from "@/lib/anomaly";

export default function AnomalySummary({ values, summary: suppliedSummary, scope = "currently displayed data" }) {
  const summary = useMemo(() => suppliedSummary ?? summarizeFiniteValues(values), [values, suppliedSummary]);
  const value = (number) => typeof number === "number" && Number.isFinite(number) ? `${number > 0 ? "+" : ""}${number.toFixed(2)} °C` : "N/A";
  return <section aria-label="Anomaly summary" className="rounded border border-border-subtle bg-surface-950/70 p-2 text-caption text-text-secondary">
    <div className="mb-1 font-semibold uppercase tracking-wide text-text-primary">ANOMALY SUMMARY</div>
    <div className="grid grid-cols-2 gap-x-3 gap-y-1">
      <span>Minimum</span><span className="font-mono">{value(summary.min)}</span>
      <span>Maximum</span><span className="font-mono">{value(summary.max)}</span>
      <span>Mean</span><span className="font-mono">{value(summary.mean)}</span>
      <span>Valid Cells</span><span className="font-mono">{summary.validCells.toLocaleString("en-US")}</span>
    </div>
    <div className="mt-1 text-[10px] text-text-muted">Scope: {scope}. Missing and non-finite values are excluded.</div>
  </section>;
}
