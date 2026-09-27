"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — OceanMapPlaceholder  (Phase 34A)
//
// Reserves and shapes the main interactive-map region of the dashboard:
// a variable-selector tab strip, a toolbar, a graticule-labeled map surface,
// a legend, and a coordinate readout. PHASE 34A renders layout and chrome
// only, with realistic placeholder content — the actual map rendering (a
// later phase) replaces the surface in place; everything around it is meant
// to be reused as-is.
// -----------------------------------------------------------------------------

import { useState } from "react";
import { Layers, Maximize2, MousePointerClick, RotateCcw, Waves, ZoomIn, ZoomOut } from "lucide-react";
import { Badge, Button, Panel, Tabs, Tooltip } from "@/components/ui";

const VARIABLE_TABS = [
  { value: "sst", label: "Sea Surface Temp." },
  { value: "sss", label: "Salinity" },
  { value: "currents", label: "Currents" },
  { value: "anomaly", label: "Anomaly" },
];

const LAT_LINES = [30, 25, 20, 15, 10, 5];
const LON_LINES = [40, 55, 70, 85, 100];

function ToolbarButton({ icon, label }) {
  return (
    <Tooltip content={label}>
      <span>
        <Button variant="ghost" size="sm" iconOnly icon={icon} aria-label={label} />
      </span>
    </Tooltip>
  );
}

export default function OceanMapPlaceholder() {
  const [variable, setVariable] = useState("sst");
  const activeLabel = VARIABLE_TABS.find((tab) => tab.value === variable)?.label;

  return (
    <Panel
      emphasis="raised"
      title="Ocean Reconstruction Map"
      subtitle="North Indian Ocean — 0.25° gridded field"
      icon={Waves}
      bodyClassName="flex flex-col gap-4"
      headerActions={
        <div className="hidden items-center gap-1 sm:flex">
          <ToolbarButton icon={Layers} label="Toggle layers" />
          <ToolbarButton icon={RotateCcw} label="Reset view" />
          <ToolbarButton icon={Maximize2} label="Fullscreen" />
        </div>
      }
    >
      <Tabs items={VARIABLE_TABS} value={variable} onChange={setVariable} id="dashboard-map-variable" />

      {/* Map surface placeholder — swapped for the real map in a later phase */}
      <div className="relative min-h-[420px] flex-1 overflow-hidden rounded-md border border-border bg-grid-subtle bg-grid lg:min-h-[480px]">
        {/* Graticule labels */}
        <div className="pointer-events-none absolute inset-y-6 left-3 flex flex-col justify-between text-caption font-mono text-text-disabled">
          {LAT_LINES.map((lat) => (
            <span key={lat}>{lat}°N</span>
          ))}
        </div>
        <div className="pointer-events-none absolute inset-x-10 bottom-2 hidden justify-between text-caption font-mono text-text-disabled sm:flex">
          {LON_LINES.map((lon) => (
            <span key={lon}>{lon}°E</span>
          ))}
        </div>

        {/* Center placeholder state */}
        <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-full border border-border-accent bg-surface-raised/60 text-accent-400 shadow-glow-sm">
            <MousePointerClick size={22} strokeWidth={1.5} aria-hidden="true" />
          </span>
          <p className="max-w-sm text-small text-text-secondary">
            Map rendering is wired up in a later phase. Once the reconstructed field is loaded,
            click anywhere on the grid to inspect a point.
          </p>
          <Badge variant="neutral" size="sm">
            {activeLabel} · No data loaded
          </Badge>
        </div>

        {/* Zoom controls */}
        <div className="absolute right-3 top-3 flex flex-col gap-1">
          <Button variant="secondary" size="sm" iconOnly icon={ZoomIn} aria-label="Zoom in" />
          <Button variant="secondary" size="sm" iconOnly icon={ZoomOut} aria-label="Zoom out" />
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-md border border-border bg-surface-base/90 px-3 py-2 backdrop-blur-sm">
          <span className="text-caption text-text-muted">18°C</span>
          <span
            className="h-2 w-24 rounded-full bg-gradient-to-r from-info via-accent-400 to-warning sm:w-28"
            aria-hidden="true"
          />
          <span className="text-caption text-text-muted">32°C</span>
        </div>
      </div>

      {/* Coordinate readout */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-border-subtle pt-3 text-caption font-mono text-text-muted">
        <span>LAT --.--°</span>
        <span>LON --.--°</span>
        <span>DEPTH -- m</span>
        <span>VALUE --</span>
      </div>
    </Panel>
  );
}