"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — route-group layout  (Phase 32E, updated Phase 34A,
// Phase 34D wired the dashboard's point-inspection workflow into this file's
// InspectionPanel slot, Phase 34E adds the dashboard's KPI drawer content and
// threads the shared reconstruction result through to <PointInspection>)
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
//
// Phase 34D: the dashboard's InspectionPanel content is now the real
// <PointInspection> (components/inspection), not a static placeholder.
// Split into an outer component that stands up PointInspectionProvider and
// an inner AppShellContent that reads it, because a component can't consume
// a context it creates in its own return statement — see
// dashboard/_context/PointInspectionContext.js for why this state has to
// live above both OceanMap (in page.js) and this file's InspectionPanel
// slot in the first place.
//
// Phase 34E: the dashboard's KPI drawer content is now the real
// <DashboardKPIs> (dashboard/_components/DashboardKPIs.js), not the static
// DASHBOARD_KPIS placeholder array this file used to hardcode — it reads
// PointInspectionContext itself, the same way <PointInspection> does, so no
// KPI values need threading through this component beyond the context
// Provider already wrapping it. <PointInspection> now also receives the
// context's shared `pointInspection` (the `reconstruct()` request state) as
// a plain prop, since that component deliberately has no import of the
// context (see its own header comment).
// -----------------------------------------------------------------------------

import { useState } from "react";
import { usePathname } from "next/navigation";
import { Compass } from "lucide-react";
import { AppShell, InspectionPanel, KPIDrawer, MainContent, Sidebar, TopNavigation } from "@/components/shell";
import { Panel, StatusIndicator } from "@/components/ui";
import { PointInspection } from "@/components/inspection";
import { PointInspectionProvider, usePointInspection } from "./dashboard/_context/PointInspectionContext";
import DashboardKPIs from "./dashboard/_components/DashboardKPIs";
import { formatLat, formatLon } from "@/lib/oceanDomain";

const MAIN_CONTENT_ID = "neer-main-content";

export default function AppRouteGroupLayout({ children }) {
  // The dashboard route's current date/depth/data-mode "view" plus its
  // selection and shared reconstruction fetch — see
  // PointInspectionContext.js's header comment. Harmless to provide on every
  // route: only the dashboard's OceanMapSection/DataContextSection/
  // PointInspection/DashboardKPIs consume it today.
  return (
    <PointInspectionProvider>
      <AppShellContent>{children}</AppShellContent>
    </PointInspectionProvider>
  );
}

function AppShellContent({ children }) {
  const pathname = usePathname();
  const isDashboard = pathname === "/dashboard";
  const { selectedPoint, clearSelection, dataMode, date, depth, pointInspection } = usePointInspection();

  // Starts open: on desktop/xl there is currently no in-page control to
  // reopen it once closed (a future page can add one via headerActions/a
  // toolbar button), so defaulting to open keeps the panel reachable.
  const [inspectorOpen, setInspectorOpen] = useState(true);

  const inspectionSubtitle = isDashboard
    ? selectedPoint
      ? `${formatLat(selectedPoint.lat)}, ${formatLon(selectedPoint.lon)}`
      : "Click the map to inspect a location"
    : "No selection";

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
            subtitle={inspectionSubtitle}
            open={inspectorOpen}
            onClose={() => setInspectorOpen(false)}
          >
            {isDashboard ? (
              <PointInspection
                selectedPoint={selectedPoint}
                date={date}
                depth={depth}
                dataMode={dataMode}
                onClose={clearSelection}
                reconstruction={pointInspection}
              />
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
          isDashboard ? (
            <DashboardKPIs />
          ) : (
            <KPIDrawer
              title="Key Indicators"
              summary={<span className="text-small text-text-muted">No metrics yet for this section</span>}
            />
          )
        }
      >
        <MainContent id={MAIN_CONTENT_ID}>{children}</MainContent>
      </AppShell>
    </>
  );
}