"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — Preview (development only)
//
// Phase 32D exercises the fuller InspectionPanel / KPIDrawer APIs — subtitle,
// header/footer actions, configurable width, controlled open/close, and the
// KPIDrawer `kpis`/`status`/`actions` convenience props with a resizable
// expanded panel. All values below are placeholders for visual QA only —
// not real NEER metrics.
//
//   < md   — Sidebar and InspectionPanel are both off-canvas drawers;
//            KPIDrawer expands as a bottom-sheet overlay with a backdrop.
//   md–lg  — Sidebar is an inline, collapsed icon rail (tap to expand);
//            InspectionPanel is still an adaptive drawer, a bit wider now;
//            KPIDrawer expands inline, with a drag handle to resize it.
//   >= lg  — Sidebar is inline, expanded by default.
//   >= xl  — InspectionPanel also goes inline, alongside the sidebar.
//
// Still not one of the real NEER pages — placeholder content only.
// -----------------------------------------------------------------------------

import { useState } from "react";
import { Activity, Compass, Gauge, RefreshCw, Sparkles } from "lucide-react";
import {
  AppShell,
  InspectionPanel,
  KPIDrawer,
  MainContent,
  Sidebar,
  TopNavigation,
} from "@/components/shell";
import { Badge, Button, Panel, StatusIndicator } from "@/components/ui";

// Mock/placeholder data only — shape mirrors what a real page would someday
// pass in, but none of these are actual NEER outputs.
const MOCK_KPIS = [
  { id: "metric-a", title: "Metric A", value: "0.42", icon: Gauge, description: "Placeholder detail" },
  { id: "metric-b", title: "Metric B", value: "96.2", unit: "%", icon: Activity, description: "Placeholder detail" },
  { id: "metric-c", title: "Metric C", value: "High", icon: Sparkles, description: "Placeholder detail" },
];

export default function ShellPreview() {
  const [inspectorOpen, setInspectorOpen] = useState(true);

  if (process.env.NODE_ENV === "production") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-bg-base px-6 text-center">
        <p className="text-body text-text-muted">The shell preview is only available in development.</p>
      </div>
    );
  }

  return (
    <AppShell
      navigation={<TopNavigation title="NEER" status={<StatusIndicator status="online" label="System nominal" />} />}
      sidebar={<Sidebar />}
      inspectionPanel={
        <InspectionPanel
          title="Inspector"
          subtitle="Placeholder selection"
          width={360}
          open={inspectorOpen}
          onClose={() => setInspectorOpen(false)}
          headerActions={<Button variant="ghost" size="sm" iconOnly icon={RefreshCw} aria-label="Refresh selection" />}
          footerActions={
            <>
              <Button variant="secondary" size="sm">
                Dismiss
              </Button>
              <Button variant="primary" size="sm">
                Apply
              </Button>
            </>
          }
        >
          <div className="flex flex-col gap-4">
            <Panel title="Placeholder selection" icon={Compass} emphasis="raised">
              <p className="text-small text-text-secondary">
                A page will render its own selection details here — a grid cell, a reconstruction
                result, model output, metadata, and so on. This panel only owns the surrounding chrome.
              </p>
            </Panel>
            <Badge variant="info">Example content</Badge>
          </div>
        </InspectionPanel>
      }
      kpiDrawer={
        <KPIDrawer
          title="Key Indicators"
          status={{ status: "online", label: "Feed live" }}
          kpis={MOCK_KPIS}
          actions={<Button variant="ghost" size="sm" icon={RefreshCw}>Refresh</Button>}
        />
      }
    >
      <MainContent>
        <div className="mx-auto flex max-w-4xl flex-col gap-4">
          <h1 className="text-h2 text-text-primary">Responsive shell preview</h1>
          <p className="text-body text-text-muted">
            Resize the window (or open dev tools&apos; device toolbar) to test each breakpoint. Expand
            the KPI drawer at the bottom and drag its grip handle to resize it on tablet/desktop; on
            mobile it becomes a bottom sheet instead. Above the {"\u201Cxl\u201D"} breakpoint, use the button
            below to toggle the inspection panel closed and open again.
          </p>
          {!inspectorOpen && (
            <div>
              <Button variant="secondary" size="sm" onClick={() => setInspectorOpen(true)}>
                Reopen inspection panel
              </Button>
            </div>
          )}
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