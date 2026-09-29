"use client";

import { usePathname } from "next/navigation";
import { useNeerContext } from "../_context/NeerContext";

const STEPS = [
  ["Dashboard", "/dashboard"], ["Reconstruction", "/reconstruction"], ["Map", "/dashboard"],
  ["Profile", "/vertical-profile"], ["Anomaly", "/hovmoller"], ["3D", "/3d-ocean"],
  ["Embeddings", "/embedding"], ["Explainability", "/explainability"],
  ["Evaluation", "/evaluation"], ["Export", "/system-status"],
];

export default function DemoControls() {
  const pathname = usePathname();
  const { demoMode, demoStatus, activateDemoMode, deactivateDemoMode, selectedPoint, pointInspection } = useNeerContext();
  const routeStep = STEPS.findIndex(([, route]) => pathname === route);
  const step = pathname === "/dashboard"
    ? pointInspection?.isSuccess ? 2 : selectedPoint ? 1 : 0
    : Math.max(0, routeStep);

  return (
    <div className="border-b border-border-subtle bg-surface-raised px-3 py-2 sm:px-5">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-2">
        <button type="button" onClick={demoMode ? deactivateDemoMode : activateDemoMode} disabled={demoStatus.state === "loading"}
          className="rounded border border-accent-500/50 bg-accent-900/30 px-3 py-1.5 text-caption font-semibold text-accent-200 disabled:opacity-60">
          {demoStatus.state === "loading" ? "Checking demo resources…" : demoMode ? "Exit SIH demo" : "Enter SIH demo"}
        </button>
        {demoStatus.state === "error" ? <span className="text-caption text-error" role="alert">Demo unavailable: {demoStatus.error}</span> : null}
      </div>
      {demoMode ? <nav aria-label="SIH demo workflow progress" className="mx-auto mt-2 flex max-w-[1600px] flex-wrap gap-1">
        {STEPS.map(([label, route], index) => <span key={`${label}-${index}`} aria-current={route === pathname && index === step ? "step" : undefined}
          className={`rounded px-2 py-1 text-[10px] ${index === step ? "bg-accent-700 text-white" : index < step ? "bg-accent-900/50 text-accent-200" : "bg-surface-900 text-text-muted"}`}>{String(index + 1).padStart(2, "0")} {label}</span>)}
      </nav> : null}
    </div>
  );
}
