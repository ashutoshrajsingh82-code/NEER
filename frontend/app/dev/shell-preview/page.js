"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — Preview (development only)
//
// Phase 32C made Sidebar, InspectionPanel and KPIDrawer responsive internally,
// so this preview needs no structural changes — resize the window (or use
// your browser's device toolbar) to see:
//   < md   — Sidebar and InspectionPanel are both off-canvas drawers;
//            KPIDrawer expands as a bottom-sheet overlay with a backdrop.
//   md–lg  — Sidebar is an inline, collapsed icon rail (tap to expand);
//            InspectionPanel is still an adaptive drawer, a bit wider now;
//            KPIDrawer expands inline.
//   >= lg  — Sidebar is inline, expanded by default.
//   >= xl  — InspectionPanel also goes inline, alongside the sidebar.
//
// Still not one of the real NEER pages — placeholder content only.
// -----------------------------------------------------------------------------

import { Compass, Gauge, Thermometer, Waves } from "lucide-react";
import {
  AppShell,
  InspectionPanel,
  KPIDrawer,
  MainContent,
  Sidebar,
  TopNavigation,
} from "@/components/shell";
import { Badge, MetricCard, Panel, StatusIndicator } from "@/components/ui";

export default function ShellPreview() {
  if (process.env.NODE_ENV === "production") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-bg-base px-6 text-center">
        <p className="text-body text-text-muted">The shell preview is only available in development.</p>
      </div>
    );
  }

  return (
    <AppShell
      navigation={<TopNavigation title="NEER" status={<StatusIndicator status="online" label="Model live" />} />}
      sidebar={<Sidebar />}
      inspectionPanel={
        <InspectionPanel title="Inspector">
          <div className="flex flex-col gap-4">
            <Panel title="Selected point" icon={Compass} emphasis="raised">
              <p className="text-small text-text-secondary">
                Placeholder content — a page will render real point details here.
              </p>
            </Panel>
            <MetricCard
              title="Sea Surface Temp"
              value="28.4"
              unit="°C"
              icon={Thermometer}
              status="online"
              description="Placeholder metric"
            />
          </div>
        </InspectionPanel>
      }
      kpiDrawer={
        <KPIDrawer
          summary={
            <>
              <span className="text-caption text-text-muted">RMSE</span>
              <span className="text-small font-mono text-text-primary">0.142</span>
              <span className="mx-2 h-4 w-px bg-border" aria-hidden="true" />
              <span className="text-caption text-text-muted">Coverage</span>
              <span className="text-small font-mono text-text-primary">96.2%</span>
              <span className="mx-2 h-4 w-px bg-border" aria-hidden="true" />
              <Badge variant="accent">Model v3</Badge>
            </>
          }
        >
          <div className="grid gap-4 sm:grid-cols-3">
            <MetricCard title="RMSE" value="0.142" icon={Gauge} description="Placeholder detail" />
            <MetricCard title="Coverage" value="96.2" unit="%" icon={Waves} description="Placeholder detail" />
            <MetricCard title="Confidence" value="High" icon={Waves} description="Placeholder detail" />
          </div>
        </KPIDrawer>
      }
    >
      <MainContent>
        <div className="mx-auto flex max-w-4xl flex-col gap-4">
          <h1 className="text-h2 text-text-primary">Responsive shell preview</h1>
          <p className="text-body text-text-muted">
            Resize the window (or open dev tools' device toolbar) to test each breakpoint. Try expanding
            the KPI drawer at the bottom below and above the tablet width — on mobile it becomes a
            draggable-looking bottom sheet with a backdrop; on tablet/desktop it expands inline instead.
          </p>
          {Array.from({ length: 8 }).map((_, index) => (
            <Panel key={index} title={`Placeholder section ${index + 1}`}>
              <p className="text-small text-text-secondary">
                Scrollable filler content to confirm the main area — not the whole page — scrolls, at
                every breakpoint.
              </p>
            </Panel>
          ))}
        </div>
      </MainContent>
    </AppShell>
  );
}