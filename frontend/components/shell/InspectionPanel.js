"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — InspectionPanel
//
// Right-hand region for contextual details (selected point info, layer
// controls, etc. — supplied by the page). Inline column that can collapse to
// a slim strip, but only from the `xl` breakpoint up — on mobile *and*
// tablet it's an off-canvas Drawer instead (Phase 32C: tablet doesn't have
// room for a persistent third column, so the panel "adapts to available
// space" by staying an adaptive overlay a bit longer than Sidebar does, and
// gets a wider max-width once there's enough room to make that worthwhile).
// -----------------------------------------------------------------------------

import { useState } from "react";
import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { cn } from "@/lib/cn";
import Drawer from "@/components/ui/overlays/Drawer";
import Button from "@/components/ui/primitives/Button";
import { useShell } from "./ShellContext";

/**
 * @param {string} title
 * @param {React.ReactNode} children
 */
export default function InspectionPanel({ title = "Inspector", children, className }) {
  const { inspectionOpen, setInspectionOpen } = useShell();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <>
      <aside
        className={cn(
          "hidden shrink-0 flex-col border-l border-border bg-surface-base neer-transition xl:flex",
          collapsed ? "w-12" : "w-80",
          className
        )}
      >
        {collapsed ? (
          <div className="flex flex-1 justify-center pt-3">
            <Button
              variant="ghost"
              size="sm"
              iconOnly
              icon={PanelRightOpen}
              aria-label={`Expand ${title}`}
              onClick={() => setCollapsed(false)}
            />
          </div>
        ) : (
          <>
            <header className="neer-divider flex items-center justify-between gap-2 px-4 py-3">
              <h3 className="truncate text-small font-semibold text-text-secondary">{title}</h3>
              <Button
                variant="ghost"
                size="sm"
                iconOnly
                icon={PanelRightClose}
                aria-label={`Collapse ${title}`}
                onClick={() => setCollapsed(true)}
              />
            </header>
            <div className="flex-1 overflow-y-auto px-4 py-4">{children}</div>
          </>
        )}
      </aside>

      {/* Wider on tablet (more room to spare) than on mobile — Drawer's own
          `max-w-sm` still applies below `sm`; these override it upward. */}
      <Drawer
        open={inspectionOpen}
        onClose={() => setInspectionOpen(false)}
        position="right"
        title={title}
        className="sm:max-w-sm md:max-w-md"
      >
        {children}
      </Drawer>
    </>
  );
}