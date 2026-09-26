// -----------------------------------------------------------------------------
// NEER Design System — LoadingSkeleton
//
// Placeholder shapes shown while real content loads. One component, several
// `variant`s, so any part of the app can show a shape matching what it will
// eventually render (a metric, a chart, a list of rows, etc.).
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";

function Block({ className }) {
  return <div className={cn("animate-pulse rounded-md bg-surface-raised", className)} />;
}

function TextSkeleton({ lines = 3 }) {
  return (
    <div className="flex flex-col gap-2">
      {Array.from({ length: lines }).map((_, index) => (
        <Block key={index} className={cn("h-3", index === lines - 1 ? "w-2/3" : "w-full")} />
      ))}
    </div>
  );
}

function CardSkeleton() {
  return (
    <div className="neer-panel flex flex-col gap-3 px-5 py-4">
      <div className="flex items-center gap-2.5">
        <Block className="h-8 w-8 shrink-0 rounded-md" />
        <Block className="h-3 w-1/3" />
      </div>
      <Block className="h-4 w-3/4" />
      <Block className="h-3 w-1/2" />
    </div>
  );
}

function MetricSkeleton() {
  return (
    <div className="neer-panel flex flex-col gap-3 px-5 py-4">
      <div className="flex items-center justify-between gap-2.5">
        <div className="flex items-center gap-2.5">
          <Block className="h-8 w-8 shrink-0 rounded-md" />
          <Block className="h-3 w-20" />
        </div>
        <Block className="h-2 w-2 rounded-full" />
      </div>
      <Block className="h-6 w-24" />
      <Block className="h-3 w-2/3" />
    </div>
  );
}

function ChartSkeleton() {
  const barHeights = [40, 65, 50, 80, 60, 90, 45];
  return (
    <div className="flex flex-1 flex-col gap-4">
      <Block className="h-3 w-1/3" />
      <div className="flex flex-1 items-end gap-2 rounded-md border border-border bg-surface-sunken/40 p-4">
        {barHeights.map((height, index) => (
          <Block key={index} className="w-full" style={{ height: `${height}%` }} />
        ))}
      </div>
    </div>
  );
}

function PanelSkeleton() {
  return (
    <div className="neer-panel flex flex-col">
      <div className="neer-divider flex items-center gap-3 px-5 py-4">
        <Block className="h-8 w-8 shrink-0 rounded-md" />
        <Block className="h-3 w-1/4" />
      </div>
      <div className="flex flex-col gap-2.5 px-5 py-4">
        <Block className="h-3 w-full" />
        <Block className="h-3 w-full" />
        <Block className="h-3 w-2/3" />
      </div>
    </div>
  );
}

function ListSkeleton({ count = 4 }) {
  return (
    <div className="flex flex-col gap-3">
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="flex items-center gap-3">
          <Block className="h-8 w-8 shrink-0 rounded-full" />
          <Block className="h-3 flex-1" />
          <Block className="h-3 w-12 shrink-0" />
        </div>
      ))}
    </div>
  );
}

const VARIANTS = {
  text: TextSkeleton,
  card: CardSkeleton,
  metric: MetricSkeleton,
  chart: ChartSkeleton,
  panel: PanelSkeleton,
  list: ListSkeleton,
};

/**
 * @param {"text"|"card"|"metric"|"chart"|"panel"|"list"} variant
 * @param {number} lines - for the "text" variant
 * @param {number} count - for the "list" variant
 * @param {string} label - accessible loading label (default "Loading")
 */
export default function LoadingSkeleton({ variant = "text", lines, count, label = "Loading", className }) {
  const Variant = VARIANTS[variant] ?? TextSkeleton;

  return (
    <div role="status" aria-busy="true" className={className}>
      <span className="sr-only">{label}</span>
      <Variant lines={lines} count={count} />
    </div>
  );
}