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
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { Compass } from "lucide-react";
import { AppShell, InspectionPanel, KPIDrawer, MainContent, Sidebar, TopNavigation } from "@/components/shell";
import { Panel } from "@/components/ui";
import { PointInspection } from "@/components/inspection";
import { DateDepthProvider } from "./dashboard/_context/DateDepthContext";
import { PointInspectionProvider } from "./dashboard/_context/PointInspectionContext";
import DashboardKPIs from "./dashboard/_components/DashboardKPIs";
import { formatLat, formatLon } from "@/lib/oceanDomain";
import { useNeerContext } from "./dashboard/_context/NeerContext";
import { formatDepth } from "@/lib/oceanDomain";
import DateControls from "./dashboard/_components/DateControls";
import DemoControls from "./dashboard/_components/DemoControls";

const MAIN_CONTENT_ID = "neer-main-content";

export default function AppRouteGroupLayout({ children }) {
  // Phase 36A: DateDepthProvider (backend-driven available/selected
  // date + depth, shared app-wide) sits above PointInspectionProvider (the
  // dashboard's selected point + shared reconstruction fetch, which
  // consumes it) — see each file's header comment. Harmless to provide on every
  // route: only the dashboard's OceanMapSection/DataContextSection/
  // PointInspection/DashboardKPIs consume it today.
  return (
    <DateDepthProvider>
      <PointInspectionProvider>
        <AppShellContent>{children}</AppShellContent>
      </PointInspectionProvider>
    </DateDepthProvider>
  );
}

function AppShellContent({ children }) {
  const pathname = usePathname();
  const isDashboard = pathname === "/dashboard";
  const { selectedPoint, clearSelection, dataMode, date, depth, pointInspection, pointStatus, modelInfoData, variableMode, setVariableMode, availableDates, availableDepths, selectDate, selectDepth, datesStatus, datesError, canStepPrev, canStepNext, stepDate, retryDates, depthsStatus, depthsError, retryDepths } = useNeerContext();

  // Starts open: on desktop/xl there is currently no in-page control to
  // reopen it once closed (a future page can add one via headerActions/a
  // toolbar button), so defaulting to open keeps the panel reachable.
  const [inspectorOpen, setInspectorOpen] = useState(true);

  const inspectionSubtitle = selectedPoint
    ? `${formatLat(selectedPoint.lat)}, ${formatLon(selectedPoint.lon)}`
    : isDashboard ? "Click the map to inspect a location" : "No location selected";

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
          <TopNavigation title="NEER" />
        }
        sidebar={<Sidebar />}
        inspectionPanel={
          <InspectionPanel
            title="Point Inspection"
            subtitle={inspectionSubtitle}
            open={inspectorOpen}
            onClose={() => setInspectorOpen(false)}
          >
            {selectedPoint ? (
              <PointInspection
                selectedPoint={selectedPoint}
                date={date}
                depth={depth}
                dataMode={dataMode}
                onClose={clearSelection}
                reconstruction={pointInspection}
                pointStatus={pointStatus}
                modelInfo={modelInfoData}
                variableMode={variableMode}
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
        <MainContent id={MAIN_CONTENT_ID}>
          <div className="sticky top-0 z-raised border-b border-border-subtle bg-surface-base/95 px-3 py-2 backdrop-blur sm:px-5">
            <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-4 gap-y-2 text-[11px]">
              <DateControls className="flex items-center gap-2 [&>p]:mb-0" availableDates={availableDates ?? []} date={date} status={datesStatus} canStepPrev={canStepPrev} canStepNext={canStepNext} onStepDate={stepDate} onSelectDate={selectDate} message={datesError?.message} onRetry={retryDates} />
              <Link href="/dashboard" title="Select location on the ocean map" className="min-w-0 text-text-muted hover:text-accent-300">
                LOCATION <span className="ml-1 font-mono text-text-primary">{selectedPoint ? `${formatLat(selectedPoint.lat)} · ${formatLon(selectedPoint.lon)}` : "Not selected"}</span>
              </Link>
              <label className="flex items-center gap-2 text-text-muted">DEPTH
                <select aria-label="Global scientific depth" value={depth ?? ""} onChange={(event) => selectDepth(Number(event.target.value))} disabled={depthsStatus !== "success"} className="rounded border border-border-subtle bg-surface-900 px-2 py-1.5 font-mono text-text-primary">
                  {depthsStatus !== "success" || depth == null ? <option value="">{depthsStatus === "loading" ? "Loading" : depthsStatus === "empty" ? "No depths" : "Unavailable"}</option> : null}
                  {(availableDepths ?? []).map((item) => <option key={item} value={item}>{formatDepth(item)}</option>)}
                </select>
              </label>
              {depthsStatus === "error" ? <button type="button" onClick={retryDepths} className="text-[10px] text-accent-300 underline" title={depthsError?.message}>Retry depths</button> : null}
              <div className="ml-auto flex items-center gap-1" role="group" aria-label="Global scientific variable mode">
                <span className="mr-1 text-text-muted">MODE</span>
                {["temperature", "anomaly"].map((mode) => <button key={mode} type="button" aria-pressed={variableMode === mode} onClick={() => setVariableMode(mode)} className={`rounded px-2 py-1.5 font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300 ${variableMode === mode ? "bg-accent-900/60 text-accent-200" : "text-text-muted hover:text-text-primary"}`}>{mode === "temperature" ? "MODEL OUTPUT" : "ANOMALY"}</button>)}
              </div>
            </div>
          </div>
          <DemoControls />
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={pathname} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -2 }} transition={{ duration: 0.14, ease: "easeOut" }} className="min-h-full">
              {children}
            </motion.div>
          </AnimatePresence>
        </MainContent>
      </AppShell>
    </>
  );
}
