"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — DepthControls  (Phase 36B)
//
// A depth slider and a row of depth chips over the SAME list of available
// depths (from the backend via DateDepthContext) and the same selected depth.
//
// The slider's positions are INDICES into that list, never raw metre values:
// NEER's levels (0, 5, 10, 20 … 500, 700, 1000 m) are very unevenly spaced, so
// a metre-valued slider would cram the shallow levels into a sliver, park the
// thumb between levels, and offer values the model doesn't have. Index
// positions make every level equally reachable and every position a real one.
//
// The chips follow the radio-group keyboard model: one tab stop on the
// selected chip; arrows move AND select, Home/End jump to the shallowest/
// deepest; Enter/Space also select. They wrap onto several lines instead of
// scrolling, so every level stays visible at any width.
// -----------------------------------------------------------------------------

import { useRef } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { Slider } from "@/components/ui";
import { depthToIndex, describeDepth, formatDepthMetres, indexToDepth, navigateDepth } from "@/lib/dateDepthControls";

/**
 * @param {number[]} depths - ascending, metres
 * @param {number|null} depth - selected depth
 * @param {"loading"|"success"|"error"} status
 * @param {(depth: number) => void} onSelectDepth - only ever called with a listed depth
 * @param {boolean} [busy] - the map is fetching the field for the selected date/depth
 * @param {string} [message] - caption when the list is a fallback / differs from expected
 */
export default function DepthControls({ depths, depth, status, onSelectDepth, busy = false, message, className }) {
  const chipRefs = useRef(new Map());
  const index = depthToIndex(depths, depth);
  const disabled = depths.length === 0 || status === "loading";

  function handleChipKeyDown(event) {
    const next = navigateDepth(depths, depth, event.key);
    if (next === null || next === depth) {
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) event.preventDefault();
      return;
    }
    event.preventDefault();
    onSelectDepth?.(next);
    chipRefs.current.get(next)?.focus();
  }

  return (
    <div className={cn("min-w-0", className)}>
      <p className="mb-1.5 flex items-center gap-2 text-caption uppercase tracking-widest text-text-muted">
        <span>Depth</span>
        {busy && <Loader2 size={11} className="animate-spin text-accent-400" aria-label="Updating map" />}
      </p>

      {status === "loading" && depths.length === 0 ? (
        <p className="text-caption text-text-muted" role="status">
          Loading depth levels…
        </p>
      ) : (
        <>
          <Slider
            label="Depth level"
            min={0}
            max={Math.max(0, depths.length - 1)}
            step={1}
            value={Math.max(0, index)}
            disabled={disabled}
            onChange={(position) => {
              const next = indexToDepth(depths, position);
              if (next !== null) onSelectDepth?.(next);
            }}
            formatValue={(position) => formatDepthMetres(depths[position])}
          />

          <div
            role="radiogroup"
            aria-label="Depth level in metres"
            onKeyDown={handleChipKeyDown}
            className="mt-3 flex flex-wrap items-center gap-1.5"
          >
            {depths.map((level) => {
              const selected = level === depth;
              return (
                <button
                  key={level}
                  ref={(node) => {
                    if (node) chipRefs.current.set(level, node);
                    else chipRefs.current.delete(level);
                  }}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  aria-label={describeDepth(level)}
                  title={level === 0 ? "Surface" : undefined}
                  // Roving tabindex: a single tab stop on the selected chip
                  // (or the first, if nothing is selected).
                  tabIndex={selected || (index === -1 && level === depths[0]) ? 0 : -1}
                  disabled={disabled}
                  onClick={() => onSelectDepth?.(level)}
                  className={cn(
                    "rounded-full border px-2.5 py-1 font-mono text-caption font-medium neer-transition",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
                    "disabled:cursor-not-allowed disabled:opacity-45",
                    selected
                      ? "border-border-accent bg-accent-900/40 text-accent-300 shadow-glow-sm"
                      : "border-border text-text-muted hover:border-border-strong hover:text-text-secondary"
                  )}
                >
                  {formatDepthMetres(level)}
                </button>
              );
            })}
          </div>
        </>
      )}

      {message && (
        <p className="mt-1.5 text-caption text-text-muted" role="status">
          {message}
        </p>
      )}
    </div>
  );
}