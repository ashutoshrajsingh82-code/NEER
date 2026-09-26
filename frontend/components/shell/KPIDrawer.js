"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — KPIDrawer  (Phase 32D)
//
// A persistent bottom strip (a compact summary row) that expands for detail.
// Generic — it renders whatever KPI cards/status/actions a page supplies; it
// has no knowledge of any specific NEER metric.
//
// Two ways to supply content:
//   - `kpis` (+ optional `status`/`actions`) — the convenience path: KPIDrawer
//     builds the collapsed summary row and an expanded MetricCard grid for you.
//   - `summary`/`children` — full manual control, same as Phase 32C, for
//     anything the convenience path can't express. Either overrides the
//     matching auto-built piece.
//
// Responsive (unchanged from Phase 32C):
//   >= md (tablet/desktop) — expands inline, pushing layout (no backdrop),
//     with a drag handle to resize how much of it shows (Phase 32D).
//   <  md (mobile)         — expands as a true bottom-sheet overlay: fixed to
//     the viewport, portaled to <body>, with a backdrop, focus trap, and
//     Escape-to-close via the shared useDialog hook.
//
// `expanded` works controlled or uncontrolled, same pattern as
// InspectionPanel's `open`: pass `expanded`/`onExpandedChange` to own it, or
// leave both out and KPIDrawer manages its own state.
// -----------------------------------------------------------------------------

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronUp, GripHorizontal, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { DURATION, EASE_OUT, fade } from "@/lib/motion";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { useDialog } from "@/lib/useDialog";
import Button from "@/components/ui/primitive/Button";
import MetricCard from "@/components/ui/data/MetricCard";
import StatusIndicator from "@/components/ui/primitive/StatusIndicator";

const DEFAULT_CONTENT_HEIGHT = 256; // px — matches the old fixed `max-h-64`
const DEFAULT_MIN_HEIGHT = 160;
const DEFAULT_MAX_HEIGHT = 480;

/** Renders whatever shape `status` was given as a consistent StatusIndicator. */
function StatusSlot({ status, size = "sm" }) {
  if (!status) return null;
  if (typeof status === "object" && !("$$typeof" in status)) {
    return <StatusIndicator status={status.status} label={status.label} size={size} />;
  }
  if (typeof status === "string") return <StatusIndicator status={status} size={size} />;
  return status; // already a rendered node (e.g. a custom <StatusIndicator />)
}

/** Compact auto-built collapsed row from `status` + the first few `kpis`. */
function DefaultSummary({ title, status, kpis }) {
  const preview = (kpis ?? []).slice(0, 4);
  if (!status && preview.length === 0) {
    return <span className="text-small text-text-muted">{title}</span>;
  }
  return (
    <>
      {status && <StatusSlot status={status} />}
      {preview.map((kpi, index) => (
        <span key={kpi.id ?? kpi.title ?? index} className="flex shrink-0 items-center gap-4">
          {(status || index > 0) && <span className="h-4 w-px bg-border" aria-hidden="true" />}
          <span className="flex items-baseline gap-1.5 whitespace-nowrap">
            <span className="text-caption text-text-muted">{kpi.title}</span>
            <span className="text-small font-mono text-text-primary">
              {kpi.value}
              {kpi.unit ? ` ${kpi.unit}` : ""}
            </span>
          </span>
        </span>
      ))}
    </>
  );
}

/** Auto-built expanded body: a MetricCard grid from `kpis`. */
function DefaultBody({ kpis }) {
  if (!kpis || kpis.length === 0) return null;
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {kpis.map((kpi, index) => (
        <MetricCard key={kpi.id ?? kpi.title ?? index} {...kpi} />
      ))}
    </div>
  );
}

/**
 * @param {string} title - shown when no `summary` is given, and as the mobile sheet's heading
 * @param {React.ReactNode} summary - always-visible collapsed content; overrides the auto-built row
 * @param {Array<object>} kpis - MetricCard prop objects; powers the default summary + body when `summary`/`children` are omitted
 * @param {React.ReactNode|{status: string, label?: string}} status - shown in the collapsed row and expanded header
 * @param {React.ReactNode} actions - buttons rendered in the expanded header (and, via context, discoverable from the collapsed row)
 * @param {React.ReactNode} children - detail content revealed when expanded; overrides the auto-built MetricCard grid
 * @param {boolean} expanded - controlled expand state (uncontrolled if omitted)
 * @param {boolean} defaultExpanded
 * @param {(expanded: boolean) => void} onExpandedChange
 * @param {boolean} resizable - whether the inline (tablet/desktop) expanded panel shows a drag handle
 * @param {number} minHeight - px, drag-resize lower bound
 * @param {number} maxHeight - px, drag-resize upper bound
 */
export default function KPIDrawer({
  title = "Key Indicators",
  summary,
  kpis,
  status,
  actions,
  children,
  expanded: expandedProp,
  defaultExpanded = false,
  onExpandedChange,
  resizable = true,
  minHeight = DEFAULT_MIN_HEIGHT,
  maxHeight = DEFAULT_MAX_HEIGHT,
  className,
}) {
  const isControlled = expandedProp !== undefined;
  const [internalExpanded, setInternalExpanded] = useState(defaultExpanded);
  const expanded = isControlled ? expandedProp : internalExpanded;

  const [contentHeight, setContentHeight] = useState(DEFAULT_CONTENT_HEIGHT);
  const dragStateRef = useRef(null);

  const isMobile = useMediaQuery("(max-width: 767px)");
  const isSheet = isMobile && expanded;
  const sheetRef = useDialog({ open: isSheet, onClose: () => setExpanded(false) });

  function setExpanded(value) {
    if (!isControlled) setInternalExpanded(value);
    onExpandedChange?.(value);
  }

  // Inline (tablet/desktop) expansion isn't a focus-trapped overlay, but it's
  // still nice for Escape to collapse it while it has focus within it.
  useEffect(() => {
    if (isMobile || !expanded) return undefined;
    function handleKeyDown(event) {
      if (event.key === "Escape") setExpanded(false);
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMobile, expanded]);

  // Drag-to-resize the inline expanded content height. Pointer capture on the
  // handle itself means we don't need document-level listeners.
  function handleResizePointerDown(event) {
    dragStateRef.current = { startY: event.clientY, startHeight: contentHeight };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function handleResizePointerMove(event) {
    if (!dragStateRef.current) return;
    const delta = dragStateRef.current.startY - event.clientY; // drag up = taller
    const next = Math.min(maxHeight, Math.max(minHeight, dragStateRef.current.startHeight + delta));
    setContentHeight(next);
  }
  function handleResizePointerUp(event) {
    dragStateRef.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  }

  const body = children ?? <DefaultBody kpis={kpis} />;

  return (
    <div className={cn("relative shrink-0 border-t border-border bg-surface-base", className)}>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-controls="neer-kpi-drawer-content"
        className={cn(
          "flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left neer-transition",
          "hover:bg-surface-raised/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
        )}
      >
        <span className="flex min-w-0 flex-1 items-center gap-4 overflow-x-auto">
          {summary ?? <DefaultSummary title={title} status={status} kpis={kpis} />}
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
              {(status || actions) && (
                <div className="neer-divider flex items-center justify-between gap-3 px-4 py-2">
                  <div className="flex min-w-0 items-center gap-3">
                    <h4 className="truncate text-small font-semibold text-text-secondary">{title}</h4>
                    <StatusSlot status={status} />
                  </div>
                  {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
                </div>
              )}

              <div className="overflow-y-auto px-4 py-4" style={{ height: contentHeight }}>
                {body}
              </div>

              {resizable && (
                <div
                  role="separator"
                  aria-orientation="horizontal"
                  aria-label={`Resize ${title}`}
                  onPointerDown={handleResizePointerDown}
                  onPointerMove={handleResizePointerMove}
                  onPointerUp={handleResizePointerUp}
                  className="flex h-3 cursor-ns-resize touch-none items-center justify-center border-t border-border-subtle"
                >
                  <GripHorizontal size={14} strokeWidth={1.75} className="text-text-disabled" aria-hidden="true" />
                </div>
              )}
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
                  <div className="neer-divider flex shrink-0 items-center justify-between gap-3 px-4 pb-2.5 pt-1.5">
                    <div className="flex min-w-0 items-center gap-3">
                      <h3 className="truncate text-small font-semibold text-text-secondary">{title}</h3>
                      <StatusSlot status={status} />
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      {actions}
                      <Button variant="ghost" size="sm" iconOnly icon={X} aria-label="Close" onClick={() => setExpanded(false)} />
                    </div>
                  </div>
                  <div className="flex-1 overflow-y-auto px-4 py-4">{body}</div>
                </motion.div>
              </>
            )}
          </AnimatePresence>,
          document.body
        )}
    </div>
  );
}