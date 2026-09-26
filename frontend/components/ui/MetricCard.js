"use client";

// -----------------------------------------------------------------------------
// NEER Design System — MetricCard
//
// Generic scientific metric display. Does not know or hardcode what metric
// it's showing (SST, SSS, RMSE, model confidence, ...) — the caller supplies
// title/value/unit and, optionally, a trend and secondary value.
// -----------------------------------------------------------------------------

import { Minus, TrendingDown, TrendingUp } from "lucide-react";
import { cn } from "@/lib/cn";
import StatusIndicator from "./StatusIndicator";

const TREND_ICONS = {
  up: TrendingUp,
  down: TrendingDown,
  flat: Minus,
};

// Default color association per direction; callers can override via
// `trend.tone` for metrics where "up" isn't necessarily good (e.g. RMSE).
const DEFAULT_TONE_BY_DIRECTION = {
  up: "positive",
  down: "negative",
  flat: "neutral",
};

const TONE_CLASSES = {
  positive: "text-success",
  negative: "text-error",
  neutral: "text-text-muted",
};

/**
 * @param {string} title
 * @param {string|number} value
 * @param {string} unit
 * @param {string} description
 * @param {React.ComponentType} icon
 * @param {{direction: "up"|"down"|"flat", value: string|number, tone?: "positive"|"negative"|"neutral"}} trend
 * @param {"online"|"processing"|"warning"|"error"|"offline"|"unknown"} status
 * @param {{label: string, value: string|number, unit?: string}} secondaryValue
 */
export default function MetricCard({
  title,
  value,
  unit,
  description,
  icon: Icon,
  trend,
  status,
  secondaryValue,
  className,
}) {
  const TrendIcon = trend ? TREND_ICONS[trend.direction] ?? Minus : null;
  const trendTone = trend ? trend.tone ?? DEFAULT_TONE_BY_DIRECTION[trend.direction] ?? "neutral" : null;

  return (
    <div className={cn("neer-panel flex flex-col gap-3 px-5 py-4", className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          {Icon && (
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface-raised text-accent-400">
              <Icon size={16} strokeWidth={1.75} aria-hidden="true" />
            </span>
          )}
          <h4 className="truncate text-small font-medium text-text-secondary">{title}</h4>
        </div>
        {status && <StatusIndicator status={status} showLabel={false} size="sm" />}
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className="text-data-lg font-mono text-text-primary">{value}</span>
        {unit && <span className="text-small text-text-muted">{unit}</span>}
      </div>

      {(description || trend) && (
        <div className="flex items-center justify-between gap-2">
          {description && <p className="truncate text-caption text-text-muted">{description}</p>}
          {trend && TrendIcon && (
            <span className={cn("flex shrink-0 items-center gap-1 text-caption font-medium", TONE_CLASSES[trendTone])}>
              <TrendIcon size={13} strokeWidth={2} aria-hidden="true" />
              {trend.value}
            </span>
          )}
        </div>
      )}

      {secondaryValue && (
        <div className="neer-divider flex items-center justify-between pt-3">
          <span className="text-caption text-text-muted">{secondaryValue.label}</span>
          <span className="text-small font-mono text-text-secondary">
            {secondaryValue.value}
            {secondaryValue.unit && <span className="ml-1 text-text-muted">{secondaryValue.unit}</span>}
          </span>
        </div>
      )}
    </div>
  );
}