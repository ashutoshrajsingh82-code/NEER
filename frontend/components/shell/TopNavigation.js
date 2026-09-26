"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — TopNavigation
//
// Sticky top bar. Derives the "current section" indicator automatically from
// the route (via NAV_ITEMS + usePathname).
//
// Responsive (Phase 32C): the menu trigger hides at `md` (Sidebar becomes
// inline there) and the inspector trigger hides at `xl` (InspectionPanel
// stays an adaptive off-canvas panel through tablet, and only goes inline at
// `xl`) — these thresholds must stay in sync with Sidebar and InspectionPanel.
// -----------------------------------------------------------------------------

import { usePathname } from "next/navigation";
import { ChevronRight, Menu, PanelRight, Waves } from "lucide-react";
import { cn } from "@/lib/cn";
import { getActiveNavItem } from "@/lib/navigation";
import Button from "@/components/ui/primitives/Button";
import { useShell } from "./ShellContext";

/**
 * @param {string} title - brand/application title next to the mark
 * @param {React.ReactNode} status - optional system status indicator (e.g. a <StatusIndicator />)
 * @param {React.ReactNode} actions - optional space for future global actions (search, notifications, profile, ...)
 * @param {React.ReactNode} centerSlot - optional extra centered content (e.g. a search box)
 */
export default function TopNavigation({ title = "NEER", status, actions, centerSlot, className }) {
  const pathname = usePathname();
  const current = getActiveNavItem(pathname);
  const CurrentIcon = current?.icon;
  const { setSidebarOpen, setInspectionOpen } = useShell();

  return (
    <header
      className={cn(
        "sticky top-0 z-raised flex h-14 shrink-0 items-center justify-between gap-4",
        "border-b border-border bg-surface-base px-4",
        className
      )}
    >
      <div className="flex min-w-0 items-center gap-3">
        <Button
          variant="ghost"
          size="sm"
          iconOnly
          icon={Menu}
          aria-label="Open navigation menu"
          className="md:hidden"
          onClick={() => setSidebarOpen(true)}
        />

        <div className="flex shrink-0 items-center gap-2">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-border-accent bg-surface-raised text-accent-400">
            <Waves size={16} strokeWidth={1.75} aria-hidden="true" />
          </span>
          <span className="truncate text-small font-semibold tracking-wide text-text-primary">{title}</span>
        </div>

        {/* Current-section indicator — derived from the route, not passed in */}
        {current && (
          <div className="hidden min-w-0 items-center gap-1.5 sm:flex" aria-hidden="true">
            <ChevronRight size={14} strokeWidth={2} className="shrink-0 text-text-disabled" />
            {CurrentIcon && <CurrentIcon size={14} strokeWidth={1.75} className="shrink-0 text-accent-400" />}
            <span className="truncate text-small text-text-secondary">{current.label}</span>
          </div>
        )}
        {current && <span className="sr-only">{`Current section: ${current.label}`}</span>}
      </div>

      {centerSlot && (
        <div className="hidden min-w-0 flex-1 items-center justify-center xl:flex">{centerSlot}</div>
      )}

      <div className="flex shrink-0 items-center gap-3">
        {status}
        {actions}
        <Button
          variant="ghost"
          size="sm"
          iconOnly
          icon={PanelRight}
          aria-label="Open inspection panel"
          className="xl:hidden"
          onClick={() => setInspectionOpen(true)}
        />
      </div>
    </header>
  );
}