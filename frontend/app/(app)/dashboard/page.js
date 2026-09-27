// -----------------------------------------------------------------------------
// NEER — /dashboard  (Phase 34A: foundation, Phase 34B: mission header +
// system/context information, Phase 34C: interactive ocean map, Phase 34D:
// point-inspection workflow, Phase 34E: full dashboard integration + QA)
//
// Renders inside the existing (app) route-group shell — AppShell/
// TopNavigation/Sidebar are mounted once in layout.js, and the shared
// InspectionPanel/KPIDrawer slots there are populated with dashboard-
// specific content whenever this route is active (see layout.js).
//
// Layout, top to bottom:
//   1. MissionHeader   — identity (NEER / SIH26066 / MoES-INCOIS) +
//                         domain facts (region, resolution, temporal)
//   2. Operational context   — one compact card holding SystemStatusPanel,
//      (single Panel)          DataContextSection (data mode / date / depth,
//                              Phase 34E: now controlled, wired to the
//                              shared context), and ModelContextPanel, so
//                              live operational state reads as one glance,
//                              not three cards
//   3. OceanMapSection — the main interactive map region (Phase 34C), via a
//                         thin client wrapper (Phase 34D) that reads the
//                         current date/depth/data-mode and selected point
//                         from PointInspectionContext, so this page (a
//                         Server Component — it exports `metadata` below)
//                         doesn't itself need to call a hook.
//
// Phase 34E completes the synchronization chain the phase spec describes —
// Date selection -> Map context -> Point selection -> Inspection panel -> KPI
// information — by making DataContextSection's date/depth controls (not just
// OceanMapSection's point selection, Phase 34D) write into
// PointInspectionContext. layout.js reads the same context to populate the
// shared InspectionPanel with <PointInspection> and the shared KPIDrawer
// slot with <DashboardKPIs> (both Phase 34E) — see
// _context/PointInspectionContext.js for why this needed a context rather
// than props threaded through this Server Component.
// -----------------------------------------------------------------------------

import { Panel } from "@/components/ui";
import OceanMapSection from "./_components/OceanMapSection";
import MissionHeader from "./_components/MissionHeader";
import SystemStatusPanel from "./_components/SystemStatusPanel";
import DataContextSection from "./_components/DataContextSection";
import ModelContextPanel from "./_components/ModelContextPanel";

export const metadata = { title: "Dashboard — NEER" };

export default function DashboardPage() {
  return (
    <div className="mx-auto flex max-w-[1600px] flex-col gap-5">
      <MissionHeader />

      <Panel emphasis="base" bodyClassName="flex flex-col gap-5 lg:flex-row lg:items-stretch lg:gap-0">
        <SystemStatusPanel state="operational" detail="All subsystems nominal" className="lg:pr-6" />
        <DataContextSection className="lg:border-l lg:border-border-subtle lg:px-6" />
        <ModelContextPanel
          version="v1.3.0"
          updatedAt="12 Mar 2024"
          className="lg:border-l lg:border-border-subtle lg:pl-6"
        />
      </Panel>

      <OceanMapSection />
    </div>
  );
}