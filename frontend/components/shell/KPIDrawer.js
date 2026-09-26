"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — KPIDrawer
//
// A persistent bottom strip (a compact summary row) that expands for detail.
//
// Responsive (Phase 32C):
//   >= md (tablet/desktop) — expands inline, pushing layout (no backdrop) —
//     there's enough vertical room that this doesn't need to cover content.
//   <  md (mobile)         — expands as a true bottom-sheet overlay: fixed to
//     the viewport, portaled to <body>, with a backdrop, focus trap, and
//     Escape-to-close via the shared useDialog hook (same one Modal/Drawer
//     use) — so it can never silently hide content behind it.
// -----------------------------------------------------------------------------

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronUp, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { DURATION, EASE_OUT, fade } from "@/lib/motion";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { useDialog } from "@/lib/useDialog";
import Button from "@/components/ui/primitives/Button";

/**
 * @param {string} title - shown when no `summary` is provided, and as the mobile sheet's heading
 * @param {React.ReactNode} summary - always-visible collapsed content, e.g. a row of key values
 * @param {React.ReactNode} children - detail content revealed when expanded
 * @param {boolean} defaultExpanded
 */
export default function KPIDrawer({ title = "Key Indicators", summary, children, defaultExpanded = false, className }) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const isMobile = useMediaQuery("(max-width: 767px)");
  const isSheet = isMobile && expanded;
  const sheetRef = useDialog({ open: isSheet, onClose: () => setExpanded(false) });

  // Inline (tablet/desktop) expansion isn't a focus-trapped overlay, but it's
  // still nice for Escape to collapse it while it has focus within it.
  useEffect(() => {
    if (isMobile || !expanded) return undefined;
    function handleKeyDown(event) {
      if (event.key === "Escape") setExpanded(false);
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isMobile, expanded]);

  return (
    <div className={cn("relative shrink-0 border-t border-border bg-surface-base", className)}>
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
        aria-controls="neer-kpi-drawer-content"
        className={cn(
          "flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left neer-transition",
          "hover:bg-surface-raised/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
        )}
      >
        <span className="flex min-w-0 flex-1 items-center gap-4 overflow-x-auto">
          {summary ?? <span className="text-small text-text-muted">{title}</span>}
        </span>
        <ChevronUp
          size={16}
          strokeWidth={1.75}
          className={cn("shrink-0 text-text-muted neer-transition", expanded && !isMobile && "rotate-180")}
          aria-hidden="true"
        />
      </button>

      {/* Tablet/desktop: expands inline, pushing layout — no backdrop needed */}
      {!isMobile && (
        <AnimatePresence initial={false}>
          {expanded && (
            <motion.div
              id="neer-kpi-drawer-content"
              key="inline-content"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1, transition: { duration: DURATION.slow, ease: EASE_OUT } }}
              exit={{ height: 0, opacity: 0, transition: { duration: DURATION.fast, ease: EASE_OUT } }}
              className="overflow-hidden border-t border-border"
            >
              <div className="max-h-64 overflow-y-auto px-4 py-4">{children}</div>
            </motion.div>
          )}
        </AnimatePresence>
      )}

      {/* Mobile: true bottom-sheet overlay, portaled to <body> so it can never
          be clipped by an ancestor and always sits above regular content. */}
      {typeof document !== "undefined" &&
        createPortal(
          <AnimatePresence>
            {isSheet && (
              <>
                <motion.div
                  className="fixed inset-0 z-modal bg-bg-base/80 backdrop-blur-sm"
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  variants={fade}
                  onClick={() => setExpanded(false)}
                  aria-hidden="true"
                />
                <motion.div
                  ref={sheetRef}
                  id="neer-kpi-drawer-content"
                  role="dialog"
                  aria-modal="true"
                  aria-label={title}
                  tabIndex={-1}
                  initial={{ y: "100%" }}
                  animate={{ y: 0, transition: { duration: DURATION.slow, ease: EASE_OUT } }}
                  exit={{ y: "100%", transition: { duration: DURATION.fast, ease: EASE_OUT } }}
                  className="fixed inset-x-0 bottom-0 z-modal flex max-h-[75vh] flex-col rounded-t-lg border-t border-border bg-surface-overlay shadow-panel"
                >
                  <div className="flex shrink-0 justify-center pt-2">
                    <span className="h-1 w-10 rounded-full bg-border-strong" aria-hidden="true" />
                  </div>
                  <div className="neer-divider flex shrink-0 items-center justify-between px-4 pb-2.5 pt-1.5">
                    <h3 className="text-small font-semibold text-text-secondary">{title}</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      iconOnly
                      icon={X}
                      aria-label="Close"
                      onClick={() => setExpanded(false)}
                    />
                  </div>
                  <div className="flex-1 overflow-y-auto px-4 py-4">{children}</div>
                </motion.div>
              </>
            )}
          </AnimatePresence>,
          document.body
        )}
    </div>
  );
}