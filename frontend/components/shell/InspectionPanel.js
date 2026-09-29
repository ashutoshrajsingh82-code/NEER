"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — InspectionPanel  (Phase 32D)
//
// Generic right-hand region for contextual details. AppShell renders one
// instance in its `inspectionPanel` slot; pages swap what's inside it
// (selected grid cell, reconstruction info, model output, metadata,
// explainability, evaluation, ...) — this component only owns the chrome:
// open/closed state, title + subtitle, header/footer actions, a scrollable
// body, a configurable width, and the desktop-column / mobile-drawer split.
// It renders no page-specific content itself (Phase 32D scope).
//
// Desktop (>= xl, matches TopNavigation/Sidebar thresholds):
//   - `open` (optional, defaults to true when uncontrolled) shows/hides the
//     column entirely, animating its width down to 0.
//   - When open, it can additionally be "collapsed" to a slim icon rail —
//     a lighter-weight dismissal than closing outright — via `collapsible`.
// Below `xl`: renders as an off-canvas Drawer overlay instead, matching
// Sidebar's adaptive pattern. Its visibility there is still driven by
// ShellContext (`inspectionOpen`), toggled from TopNavigation's trigger,
// same as Phase 32C — that contract is unchanged.
//
// Controlled vs. uncontrolled `open`: if the caller never passes `open`, the
// panel manages its own visibility (so the close button still works out of
// the box); if the caller passes `open`, they own it and must handle
// `onClose` themselves. Same pattern used by KPIDrawer for `expanded`.
// -----------------------------------------------------------------------------

import { useState } from "react";
import { motion } from "framer-motion";
import { PanelRightClose, PanelRightOpen, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { DURATION, EASE_OUT } from "@/lib/motion";
import Drawer from "@/components/ui/layout/Drawer";
import Button from "@/components/ui/primitive/Button";
import { useShell } from "./ShellContext";

const DEFAULT_WIDTH = 360; // px
const RAIL_WIDTH = 48; // px — collapsed icon-rail width

/**
 * @param {boolean} open - controls whether the desktop column renders at all (uncontrolled if omitted)
 * @param {() => void} onClose - called when the close button is pressed (and on mobile Drawer dismissal)
 * @param {string} title
 * @param {string} subtitle
 * @param {React.ReactNode} headerActions - extra controls rendered in the header, before collapse/close
 * @param {React.ReactNode} footerActions - rendered in a pinned footer bar below the scrollable body
 * @param {number} width - desktop column width in px when expanded (default 360)
 * @param {boolean} collapsible - whether the desktop column can shrink to an icon rail (default true)
 * @param {boolean} closable - whether to show the close button at all (default true)
 * @param {React.ReactNode} children - scrollable body content, supplied by the page
 */
export default function InspectionPanel({
  open,
  onClose,
  title = "Inspector",
  subtitle,
  headerActions,
  footerActions,
  width = DEFAULT_WIDTH,
  collapsible = true,
  closable = true,
  children,
  className,
}) {
  const { inspectionOpen, setInspectionOpen } = useShell();
  const [collapsed, setCollapsed] = useState(false);

  // Uncontrolled fallback so the panel is usable (and its close button does
  // something) even if a caller doesn't wire up `open`/`onClose` themselves.
  const [internalOpen, setInternalOpen] = useState(true);
  const isControlled = open !== undefined;
  const isOpen = isControlled ? Boolean(open || inspectionOpen) : internalOpen;

  function handleClose() {
    setInspectionOpen(false);
    onClose?.();
    if (!isControlled) setInternalOpen(false);
  }

  return (
    <>
      {/* Desktop / large tablet: inline column, animates width for both the
          open<->closed and expanded<->collapsed transitions. */}
      <motion.aside
        animate={{ width: isOpen ? (collapsed ? RAIL_WIDTH : width) : 0 }}
        transition={{ duration: DURATION.base, ease: EASE_OUT }}
        aria-hidden={!isOpen}
        className={cn(
          "hidden shrink-0 flex-col overflow-hidden border-border bg-surface-base xl:flex",
          isOpen && "border-l",
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
            <header className="neer-divider flex shrink-0 items-start justify-between gap-3 px-4 py-3">
              <div className="min-w-0">
                <h3 className="truncate text-small font-semibold text-text-secondary">{title}</h3>
                {subtitle && <p className="mt-0.5 truncate text-caption text-text-muted">{subtitle}</p>}
              </div>
              <div className="flex shrink-0 items-center gap-1">
                {headerActions}
                {collapsible && (
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={PanelRightClose}
                    aria-label={`Collapse ${title}`}
                    onClick={() => setCollapsed(true)}
                  />
                )}
                {closable && (
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={X}
                    aria-label={`Close ${title}`}
                    onClick={handleClose}
                  />
                )}
              </div>
            </header>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">{children}</div>

            {footerActions && (
              <footer className="neer-divider shrink-0 bg-surface-sunken/40 px-4 py-3">
                <div className="flex items-center justify-end gap-2">{footerActions}</div>
              </footer>
            )}
          </>
        )}
      </motion.aside>

      {/* Mobile + small tablet: off-canvas overlay drawer. Visibility here
          stays keyed to ShellContext, same as Phase 32C, so TopNavigation's
          trigger keeps working regardless of the desktop `open`/`onClose`
          contract above. Wider on tablet than on mobile. */}
      <Drawer
        open={inspectionOpen}
        onClose={() => {
          setInspectionOpen(false);
          onClose?.();
        }}
        position="right"
        title={title}
        subtitle={subtitle}
        headerActions={headerActions}
        footer={
          footerActions && <div className="flex items-center justify-end gap-2">{footerActions}</div>
        }
        className="sm:max-w-sm md:max-w-md"
      >
        {children}
      </Drawer>
    </>
  );
}
