"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — CalendarPopover  (Phase 36B)
//
// A month-view date picker restricted to the dates the backend lists (GET
// /dates, via DateDepthContext). Unavailable days are shown but cannot be
// selected; the month navigation cannot leave the range that has data.
//
// Rendered through a portal with fixed positioning: the controls sit inside a
// Panel that is `overflow-hidden`, which would clip an absolutely-positioned
// popover.
//
// Keyboard: arrows move by day/week, Home/End to week start/end, PageUp/
// PageDown by month (Shift = year), Enter/Space select, Escape closes and
// returns focus to the trigger. Focus can rest on an unavailable day (it is
// announced as unavailable) so arrow movement stays predictable.
// -----------------------------------------------------------------------------

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";
import {
  MONTH_NAMES,
  WEEKDAY_LABELS,
  buildMonthGrid,
  canShiftMonth,
  clampMonth,
  countAvailableInMonth,
  initialFocusDate,
  initialMonth,
  monthOf,
  monthRange,
  moveCalendarFocus,
  shiftMonth,
  yearsWithDates,
} from "@/lib/dateDepthControls";
import { formatDate } from "@/lib/format";

const POPOVER_WIDTH = 288;

/**
 * @param {string[]} availableDates - ascending "YYYY-MM-DD" list from the backend
 * @param {string|null} selectedDate
 * @param {(date: string) => void} onSelect - only ever called with an available date
 * @param {() => void} onClose
 * @param {React.RefObject<HTMLElement>} anchorRef - the trigger the popover is placed under
 */
export default function CalendarPopover({ availableDates, selectedDate, onSelect, onClose, anchorRef }) {
  const availableSet = useMemo(() => new Set(availableDates), [availableDates]);
  const range = useMemo(() => monthRange(availableDates), [availableDates]);
  const years = useMemo(() => yearsWithDates(availableDates), [availableDates]);

  const startMonth = useMemo(() => initialMonth(selectedDate, availableDates), [selectedDate, availableDates]);
  const [month, setMonth] = useState(() => clampMonth(startMonth ?? { year: 1970, month: 1 }, range));
  const [focusIso, setFocusIso] = useState(() => initialFocusDate(selectedDate, month, availableSet));
  const [position, setPosition] = useState(null);
  const popoverRef = useRef(null);
  const shouldFocusRef = useRef(true);

  // Place under the trigger (flip above if there isn't room), clamped to the viewport.
  const reposition = useCallback(() => {
    const anchor = anchorRef.current;
    if (!anchor) return;
    const rect = anchor.getBoundingClientRect();
    const height = popoverRef.current?.offsetHeight ?? 340;
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - POPOVER_WIDTH - 8));
    const below = rect.bottom + 6;
    const top = below + height > window.innerHeight - 8 && rect.top - height - 6 > 8 ? rect.top - height - 6 : below;
    setPosition({ top, left });
  }, [anchorRef]);

  useLayoutEffect(() => {
    reposition();
  }, [reposition, month]);

  useEffect(() => {
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  }, [reposition]);

  // Close on outside press / Escape.
  useEffect(() => {
    function onPointerDown(event) {
      if (popoverRef.current?.contains(event.target) || anchorRef.current?.contains(event.target)) return;
      onClose();
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [onClose, anchorRef]);

  // Move DOM focus to the roving-tabindex day whenever it changes by keyboard/open.
  useEffect(() => {
    if (!shouldFocusRef.current) return;
    popoverRef.current?.querySelector(`[data-iso="${focusIso}"]`)?.focus();
  }, [focusIso, month]);

  const weeks = useMemo(() => buildMonthGrid(month, availableSet, selectedDate), [month, availableSet, selectedDate]);
  const availableInMonth = useMemo(() => countAvailableInMonth(month, availableSet), [month, availableSet]);

  function goToMonth(next, { keepFocus = false } = {}) {
    const clamped = clampMonth(next, range);
    shouldFocusRef.current = keepFocus;
    setMonth(clamped);
    if (!keepFocus) setFocusIso(initialFocusDate(selectedDate, clamped, availableSet));
  }

  function handleDayKeyDown(event, cell) {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (cell.available) onSelect(cell.iso);
      return;
    }
    const next = moveCalendarFocus(cell.iso, event.key, range, event.shiftKey);
    if (next === cell.iso) {
      if (event.key.startsWith("Arrow") || event.key.startsWith("Page") || event.key === "Home" || event.key === "End") {
        event.preventDefault();
      }
      return;
    }
    event.preventDefault();
    shouldFocusRef.current = true;
    setFocusIso(next);
    const target = monthOf(next);
    if (target.year !== month.year || target.month !== month.month) setMonth(target);
  }

  if (!range || !startMonth) return null;

  const monthLabel = `${MONTH_NAMES[month.month - 1]} ${month.year}`;

  return createPortal(
    <div
      ref={popoverRef}
      role="dialog"
      aria-label="Choose an available date"
      style={{
        position: "fixed",
        top: position?.top ?? -9999,
        left: position?.left ?? -9999,
        width: POPOVER_WIDTH,
        visibility: position ? "visible" : "hidden",
      }}
      className="z-overlay rounded-lg border border-border-strong bg-surface-overlay p-3 shadow-raised"
    >
      {/* Month / year navigation — cannot leave the range that has data */}
      <div className="mb-2 flex items-center justify-between gap-1">
        <button
          type="button"
          aria-label="Previous month"
          disabled={!canShiftMonth(month, -1, range)}
          onClick={() => goToMonth(shiftMonth(month, -1))}
          className="flex h-7 w-7 items-center justify-center rounded-md text-text-secondary neer-transition hover:bg-surface-raised hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent"
        >
          <ChevronLeft size={16} strokeWidth={1.75} aria-hidden="true" />
        </button>

        <div className="flex items-center gap-1.5">
          <span className="text-small font-medium text-text-primary" aria-live="polite">
            {MONTH_NAMES[month.month - 1]}
          </span>
          {years.length > 1 ? (
            <select
              aria-label="Year"
              value={month.year}
              onChange={(event) => goToMonth({ year: Number(event.target.value), month: month.month })}
              className="rounded-md border border-border bg-surface-raised px-1.5 py-0.5 font-mono text-small text-text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
            >
              {years.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
          ) : (
            <span className="font-mono text-small text-text-primary">{month.year}</span>
          )}
        </div>

        <button
          type="button"
          aria-label="Next month"
          disabled={!canShiftMonth(month, 1, range)}
          onClick={() => goToMonth(shiftMonth(month, 1))}
          className="flex h-7 w-7 items-center justify-center rounded-md text-text-secondary neer-transition hover:bg-surface-raised hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent"
        >
          <ChevronRight size={16} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </div>

      <table role="grid" aria-label={monthLabel} className="w-full border-separate border-spacing-y-0.5 text-center">
        <thead>
          <tr>
            {WEEKDAY_LABELS.map((label) => (
              <th key={label} scope="col" className="pb-1 text-caption font-normal uppercase text-text-muted">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {weeks.map((week, weekIndex) => (
            <tr key={weekIndex}>
              {week.map((cell, dayIndex) => (
                <td key={dayIndex} role="gridcell" aria-selected={cell?.selected || undefined} className="p-0">
                  {cell && (
                    <button
                      type="button"
                      data-iso={cell.iso}
                      tabIndex={cell.iso === focusIso ? 0 : -1}
                      aria-disabled={!cell.available}
                      aria-current={cell.selected ? "date" : undefined}
                      aria-label={`${formatDate(cell.iso)}${cell.available ? "" : " — no data available"}`}
                      title={cell.available ? undefined : "No data for this date"}
                      onClick={() => cell.available && onSelect(cell.iso)}
                      onKeyDown={(event) => handleDayKeyDown(event, cell)}
                      onFocus={() => setFocusIso(cell.iso)}
                      className={cn(
                        "mx-auto flex h-8 w-8 items-center justify-center rounded-md font-mono text-small neer-transition",
                        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
                        cell.selected
                          ? "border border-border-accent bg-accent-500 font-medium text-text-inverse shadow-glow-sm"
                          : cell.available
                            ? "border border-border-subtle bg-surface-raised text-text-primary hover:border-border-accent hover:bg-accent-900/40 hover:text-accent-300"
                            : "cursor-not-allowed border border-transparent text-text-disabled line-through decoration-text-disabled/50"
                      )}
                    >
                      {cell.day}
                    </button>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <p className="mt-2 border-t border-border-subtle pt-2 text-caption text-text-muted" role="status">
        {availableInMonth
          ? `${availableInMonth} date${availableInMonth === 1 ? "" : "s"} with data in ${monthLabel}`
          : `No data in ${monthLabel}`}
      </p>
    </div>,
    document.body
  );
}