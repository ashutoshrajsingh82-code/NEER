"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — route-group layout  (Phase 32E)
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
// InspectionPanel and KPIDrawer are optional per AppShell's contract — they
// are included here, generic and content-free, purely so their open/closed
// and expand/collapse state can be exercised on every route while future
// phases decide, page by page, whether/what to put inside them.
// -----------------------------------------------------------------------------

import { useState } from "react";
import { Compass } from "lucide-react";
import { AppShell, InspectionPanel, KPIDrawer, MainContent, Sidebar, TopNavigation } from "@/components/shell";
import { Panel, StatusIndicator } from "@/components/ui";

const MAIN_CONTENT_ID = "neer-main-content";

export default function AppRouteGroupLayout({ children }) {
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
            title="Inspector"
            subtitle="No selection"
            open={inspectorOpen}
            onClose={() => setInspectorOpen(false)}
          >
            <Panel emphasis="raised" icon={Compass} title="Nothing selected">
              <p className="text-small text-text-secondary">
                Pages built in later phases will populate this panel with contextual detail for
                whatever is selected (a grid cell, a reconstruction result, model output, and so
                on). It has no content of its own yet.
              </p>
            </Panel>
          </InspectionPanel>
        }
        kpiDrawer={<KPIDrawer title="Key Indicators" summary={<span className="text-small text-text-muted">No metrics yet for this section</span>} />}
      >
        <MainContent id={MAIN_CONTENT_ID}>{children}</MainContent>
      </AppShell>
    </>
  );
}