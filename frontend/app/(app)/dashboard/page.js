// -----------------------------------------------------------------------------
// NEER — /dashboard  (Phase 34A: foundation, Phase 34B: mission header +
// system/context information)
//
// Renders inside the existing (app) route-group shell — AppShell/
// TopNavigation/Sidebar are mounted once in layout.js, and the shared
// InspectionPanel/KPIDrawer slots there are populated with dashboard-
// specific content whenever this route is active (see layout.js).
//
// Phase 34B layout, top to bottom:
//   1. MissionHeader        — identity (NEER / SIH26066 / MoES-INCOIS) +
//                              domain facts (region, resolution, temporal)
//   2. Operational context   — one compact card holding SystemStatusPanel,
//      (single Panel)          DataContextPanel (data mode / date / depth),
//                              and ModelContextPanel, so live operational
//                              state reads as one glance, not three cards
//   3. OceanMapPlaceholder  — main map area (Phase 34A; still a placeholder)
//
// The ocean map itself and final point-inspection behavior are not
// implemented yet — see OceanMapPlaceholder and layout.js's
// PointInspectionPlaceholder.
// -----------------------------------------------------------------------------

import { Panel } from "@/components/ui";
import MissionHeader from "./_components/MissionHeader";
import SystemStatusPanel from "./_components/SystemStatusPanel";
import DataContextPanel from "./_components/DataContextPanel";
import ModelContextPanel from "./_components/ModelContextPanel";
import OceanMapPlaceholder from "./_components/OceanMapPlaceholder";

export const metadata = { title: "Dashboard — NEER" };

export default function DashboardPage() {
  return (
    <div className="mx-auto flex max-w-[1600px] flex-col gap-5">
      <MissionHeader />

      <Panel emphasis="base" bodyClassName="flex flex-col gap-5 lg:flex-row lg:items-stretch lg:gap-0">
        <SystemStatusPanel state="operational" detail="All subsystems nominal" className="lg:pr-6" />
        <DataContextPanel
          dataMode="reconstructed"
          className="lg:border-l lg:border-border-subtle lg:px-6"
        />
        <ModelContextPanel
          version="v1.3.0"
          updatedAt="12 Mar 2024"
          className="lg:border-l lg:border-border-subtle lg:pl-6"
        />
      </Panel>

      <OceanMapPlaceholder />
    </div>
  );
}