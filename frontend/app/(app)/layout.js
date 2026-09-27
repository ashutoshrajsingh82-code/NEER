"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — route-group layout  (Phase 32E, updated Phase 34A)
//
// This is the single integration point between the generic shell
// (components/shell) and Next.js routing: every real NEER section
// (/dashboard, /reconstruction, /vertical-profile, /hovmoller, /3d-ocean,
// /embedding, /explainability, /evaluation, /data-quality, /system-status)
// lives inside the `(app)` route group and therefore shares this layout, so
// AppShell/TopNavigation/Sidebar/InspectionPanel/KPIDrawer are mounted once
// and persist across client-side navigation instead of remounting per page.
//
// AppShell itself stays free of NEER-specific content (Phase 32 requirement)
// — everything below is generic wiring + placeholder chrome. Individual
// pages (app/(app)/<route>/page.js) render only inside <MainContent>'s
// children; none of them need to know the shell exists.
//
// InspectionPanel and KPIDrawer are optional per AppShell's contract and are
// shared across every route. Phase 34A gives the dashboard route ("right
// point-inspection area" + "bottom KPI drawer" in its spec) real, dashboard-
// shaped placeholder content in those same shared slots — keyed off the
// current pathname — while every other route keeps the generic, content-free
// placeholder from Phase 32E until its own phase defines what belongs there.
// -----------------------------------------------------------------------------

import { useState } from "react";
import { usePathname } from "next/navigation";
import { Compass, Gauge, Grid3x3, Satellite, Thermometer } from "lucide-react";
import { AppShell, InspectionPanel, KPIDrawer, MainContent, Sidebar, TopNavigation } from "@/components/shell";
import { Panel, StatusIndicator } from "@/components/ui";
import PointInspectionPlaceholder from "./dashboard/_components/PointInspectionPlaceholder";

const MAIN_CONTENT_ID = "neer-main-content";

// Placeholder KPI values for the dashboard route's KPI drawer — realistic
// shape and units, wired to a live source in a later phase.
const DASHBOARD_KPIS = [
  {
    id: "coverage",
    title: "Grid Coverage",
    value: "94.2",
    unit: "%",
    icon: Grid3x3,
    trend: { direction: "up", value: "+1.2%" },
  },
  {
    id: "mean-sst",
    title: "Mean SST",
    value: "28.4",
    unit: "°C",
    icon: Thermometer,
  },
  {
    id: "argo-floats",
    title: "Active Argo Floats",
    value: "182",
    icon: Satellite,
    trend: { direction: "flat", value: "0" },
  },
  {
    id: "latency",
    title: "Reconstruction Latency",
    value: "340",
    unit: "ms",
    icon: Gauge,
    trend: { direction: "down", value: "-18ms", tone: "positive" },
  },
];

export default function AppRouteGroupLayout({ children }) {
  const pathname = usePathname();
  const isDashboard = pathname === "/dashboard";

  // Starts open: on desktop/xl there is currently no in-page control to
  // reopen it once closed (a future page can add one via headerActions/a
  // toolbar button), so defaulting to open keeps the panel reachable.
  const [inspectorOpen, setInspectorOpen] = useState(true);

  return (
    <>
      {/* Visually hidden until focused — first tab stop, lets keyboard users
          jump past the top bar/sidebar straight to page content. */}
      <a
        href={`#${MAIN_CONTENT_ID}`}
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-toast focus:rounded-md focus:bg-accent-500 focus:px-4 focus:py-2 focus:text-small focus:font-medium focus:text-text-inverse"
      >
        Skip to main content
      </a>

      <AppShell
        navigation={
          <TopNavigation title="NEER" status={<StatusIndicator status="online" label="System nominal" />} />
        }
        sidebar={<Sidebar />}
        inspectionPanel={
          <InspectionPanel
            title={isDashboard ? "Point Inspection" : "Inspector"}
            subtitle={isDashboard ? "Click the map to inspect a location" : "No selection"}
            open={inspectorOpen}
            onClose={() => setInspectorOpen(false)}
          >
            {isDashboard ? (
              <PointInspectionPlaceholder />
            ) : (
              <Panel emphasis="raised" icon={Compass} title="Nothing selected">
                <p className="text-small text-text-secondary">
                  Pages built in later phases will populate this panel with contextual detail for
                  whatever is selected (a grid cell, a reconstruction result, model output, and so
                  on). It has no content of its own yet.
                </p>
              </Panel>
            )}
          </InspectionPanel>
        }
        kpiDrawer={
          <KPIDrawer
            title={isDashboard ? "Mission KPIs" : "Key Indicators"}
            kpis={isDashboard ? DASHBOARD_KPIS : undefined}
            status={isDashboard ? { status: "online", label: "Pipeline nominal" } : undefined}
            summary={
              isDashboard ? undefined : (
                <span className="text-small text-text-muted">No metrics yet for this section</span>
              )
            }
          />
        }
      >
        <MainContent id={MAIN_CONTENT_ID}>{children}</MainContent>
      </AppShell>
    </>
  );
}