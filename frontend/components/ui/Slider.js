"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Slider
//
// A labeled range control for scientific parameters (depth, latitude,
// longitude, date/time offsets, thresholds, model parameters). The caller
// supplies min/max/step/value and an optional `formatValue` function so the
// same component can display "1,240 m", "12.34° N", or a formatted date
// without the Slider itself knowing anything about the domain.
// -----------------------------------------------------------------------------

import { useId } from "react";
import { cn } from "@/lib/cn";

/**
 * @param {number} min
 * @param {number} max
 * @param {number} step
 * @param {number} value
 * @param {(value: number) => void} onChange
 * @param {string} label
 * @param {string} unit - appended to the displayed value, e.g. "m", "°C"
 * @param {boolean} disabled
 * @param {(value: number) => string} formatValue - overrides the default "value unit" display
 */
export default function Slider({
  min = 0,
  max = 100,
  step = 1,
  value,
  onChange,
  label,
  unit,
  disabled = false,
  formatValue,
  className,
}) {
  const id = useId();
  const percent = max === min ? 0 : ((value - min) / (max - min)) * 100;
  const display = formatValue ? formatValue(value) : `${value}${unit ? ` ${unit}` : ""}`;

  return (
    <div className={cn("w-full", disabled && "opacity-45", className)}>
      {(label || display) && (
        <div className="mb-2 flex items-center justify-between gap-2">
          {label && (
            <label htmlFor={id} className="text-small font-medium text-text-secondary">
              {label}
            </label>
          )}
          <span className="text-data font-mono text-accent-300">{display}</span>
        </div>
      )}

      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange?.(Number(event.target.value))}
        style={{
          backgroundImage: `linear-gradient(90deg, var(--color-accent) ${percent}%, rgba(148,197,224,0.14) ${percent}%)`,
        }}
        className={cn(
          "h-1.5 w-full cursor-pointer appearance-none rounded-full bg-border outline-none",
          "focus-visible:ring-2 focus-visible:ring-accent-400 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-base",
          "[&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none",
          "[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-bg-base",
          "[&::-webkit-slider-thumb]:bg-accent-400 [&::-webkit-slider-thumb]:shadow-glow-sm",
          "[&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-110",
          "[&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full",
          "[&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-bg-base [&::-moz-range-thumb]:bg-accent-400",
          disabled && "cursor-not-allowed"
        )}
      />

      <div className="mt-1 flex justify-between text-caption text-text-muted">
        <span>{formatValue ? formatValue(min) : min}</span>
        <span>{formatValue ? formatValue(max) : max}</span>
      </div>
    </div>
  );
}