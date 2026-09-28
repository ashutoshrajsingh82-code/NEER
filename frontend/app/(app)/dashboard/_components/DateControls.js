"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — DateControls  (Phase 36B)
//
// Previous / calendar / next. Pure presentation over the shared date state
// (DateDepthContext): it dispatches `onSelectDate` / `onStepDate` and shows
// whatever `date` it is given. Previous/next move through the AVAILABLE
// backend dates (not ±1 calendar day) and are disabled at the first/last
// date, while dates are loading, and when there are none.
// -----------------------------------------------------------------------------

import { useCallback, useRef, useState } from "react";
import { CalendarDays, ChevronDown, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button } from "@/components/ui";
import { formatDate } from "@/lib/format";
import CalendarPopover from "./CalendarPopover";

/**
 * @param {string[]} availableDates
 * @param {string|null} date - selected "YYYY-MM-DD"
 * @param {"loading"|"success"|"empty"|"error"} status
 * @param {boolean} canStepPrev
 * @param {boolean} canStepNext
 * @param {(delta: number) => void} onStepDate
 * @param {(date: string) => boolean|void} onSelectDate
 * @param {boolean} [busy] - the map is fetching the field for the selected date/depth
 * @param {string} [message] - caption for the empty/error states
 * @param {() => void} [onRetry]
 */
export default function DateControls({
  availableDates,
  date,
  status,
  canStepPrev,
  canStepNext,
  onStepDate,
  onSelectDate,
  busy = false,
  message,
  onRetry,
  className,
}) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef(null);

  const loading = status === "loading";
  const unavailable = status === "empty" || status === "error";
  const canOpen = status === "success" && availableDates.length > 0 && !loading;

  const close = useCallback(() => {
    setOpen(false);
    triggerRef.current?.focus();
  }, []);

  function select(next) {
    onSelectDate?.(next);
    close();
  }

  const label = date
    ? formatDate(date)
    : loading
      ? "Loading dates…"
      : status === "empty"
        ? "No dates available"
        : "Dates unavailable";

  return (
    <div className={cn("min-w-0", className)}>
      <p className="mb-1.5 flex items-center gap-2 text-caption uppercase tracking-widest text-text-muted">
        <span>Date</span>
        {busy && <Loader2 size={11} className="animate-spin text-accent-400" aria-label="Updating map" />}
      </p>

      <div className="flex items-center gap-1" role="group" aria-label="Date selection">
        <Button
          variant="secondary"
          size="sm"
          iconOnly
          icon={ChevronLeft}
          aria-label="Previous available date"
          title="Previous available date"
          disabled={!canStepPrev || loading}
          onClick={() => onStepDate?.(-1)}
        />

        <button
          ref={triggerRef}
          type="button"
          aria-haspopup="dialog"
          aria-expanded={open}
          aria-label={date ? `Selected date ${formatDate(date)}. Open calendar` : label}
          disabled={!canOpen}
          onClick={() => setOpen((v) => !v)}
          className={cn(
            "flex h-8 min-w-[10.5rem] items-center justify-between gap-2 rounded-md border px-3 font-mono text-small neer-transition",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
            canOpen
              ? "border-border-strong bg-surface-raised text-text-primary hover:border-border-accent hover:bg-surface-overlay"
              : "cursor-not-allowed border-border bg-surface-sunken text-text-disabled"
          )}
        >
          <span className="flex items-center gap-2">
            <CalendarDays size={14} strokeWidth={1.75} className="text-accent-400" aria-hidden="true" />
            <span className="whitespace-nowrap">{label}</span>
          </span>
          <ChevronDown size={14} strokeWidth={1.75} aria-hidden="true" className={cn("neer-transition", open && "rotate-180")} />
        </button>

        <Button
          variant="secondary"
          size="sm"
          iconOnly
          icon={ChevronRight}
          aria-label="Next available date"
          title="Next available date"
          disabled={!canStepNext || loading}
          onClick={() => onStepDate?.(1)}
        />
      </div>

      {unavailable && (
        <p className="mt-1.5 flex items-center gap-2 text-caption text-text-muted" role="status">
          <span className="truncate">{message}</span>
          {status === "error" && onRetry && (
            <button type="button" onClick={onRetry} className="shrink-0 text-accent-400 underline">
              Retry
            </button>
          )}
        </p>
      )}

      {open && canOpen && (
        <CalendarPopover
          availableDates={availableDates}
          selectedDate={date}
          onSelect={select}
          onClose={close}
          anchorRef={triggerRef}
        />
      )}
    </div>
  );
}