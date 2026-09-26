"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — AppShell
//
// Controls the overall page layout: a fixed-height viewport with a sticky top
// bar, a left sidebar and right inspection panel flanking an independently
// scrollable main content area, and a bottom KPI drawer. AppShell itself
// contains no NEER-specific content — every region is passed in by the
// caller, so any future page can wrap itself in the same shell.
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";
import { ShellProvider } from "./ShellContext";

/**
 * @param {React.ReactNode} navigation - typically a <TopNavigation />
 * @param {React.ReactNode} sidebar - typically a <Sidebar />
 * @param {React.ReactNode} inspectionPanel - typically an <InspectionPanel />
 * @param {React.ReactNode} kpiDrawer - typically a <KPIDrawer />
 * @param {React.ReactNode} children - the main content, typically a <MainContent />
 */
export default function AppShell({ navigation, sidebar, inspectionPanel, kpiDrawer, children, className }) {
  return (
    <ShellProvider>
      <div className={cn("flex h-screen w-full flex-col overflow-hidden bg-bg-base text-text-primary", className)}>
        {navigation}

        <div className="flex flex-1 overflow-hidden">
          {sidebar}

          {/* This column, not <main> itself, owns the flex sizing — MainContent
              owns the actual scrolling element and its own overflow-y-auto. */}
          <div className="flex min-w-0 flex-1 flex-col overflow-hidden">{children}</div>

          {inspectionPanel}
        </div>

        {kpiDrawer}
      </div>
    </ShellProvider>
  );
}