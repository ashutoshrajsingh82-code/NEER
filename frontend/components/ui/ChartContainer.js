"use client";

// -----------------------------------------------------------------------------
// NEER Design System — ChartContainer
//
// Provides layout/chrome around a chart: title, subtitle, toolbar, legend
// area, loading/empty states, and an optional fullscreen mode. It does NOT
// implement any specific charting library — `children` is whatever chart the
// caller renders (SVG, canvas, a chart library component, etc.).
// -----------------------------------------------------------------------------

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, Maximize2, Minimize2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { scaleIn } from "@/lib/motion";
import Button from "./Button";
import LoadingSkeleton from "./LoadingSkeleton";

/**
 * @param {string} title
 * @param {string} subtitle
 * @param {React.ReactNode} children - the chart itself
 * @param {React.ReactNode} legend - optional legend content, rendered below the chart
 * @param {React.ReactNode} actions - optional toolbar controls, rendered top-right
 * @param {boolean} loading
 * @param {boolean} empty
 * @param {string} emptyMessage
 * @param {boolean} allowFullscreen - shows a fullscreen toggle button
 */
export default function ChartContainer({
  title,
  subtitle,
  children,
  legend,
  actions,
  loading = false,
  empty = false,
  emptyMessage = "No data available for the current selection.",
  allowFullscreen = false,
  className,
}) {
  const [fullscreen, setFullscreen] = useState(false);

  const body = loading ? (
    <LoadingSkeleton variant="chart" />
  ) : empty ? (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 py-12 text-center">
      <BarChart3 size={28} strokeWidth={1.5} className="text-text-disabled" aria-hidden="true" />
      <p className="text-small text-text-muted">{emptyMessage}</p>
    </div>
  ) : (
    children
  );

  const content = (
    <div
      className={cn(
        "neer-panel flex flex-col",
        fullscreen ? "fixed inset-4 z-modal shadow-panel" : "relative",
        className
      )}
    >
      {(title || subtitle || actions || allowFullscreen) && (
        <header className="neer-divider flex items-start justify-between gap-4 px-5 py-4">
          <div className="min-w-0">
            {title && <h3 className="truncate text-h3 text-text-primary">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-small text-text-muted">{subtitle}</p>}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {actions}
            {allowFullscreen && (
              <Button
                variant="ghost"
                size="sm"
                iconOnly
                icon={fullscreen ? Minimize2 : Maximize2}
                aria-label={fullscreen ? "Exit fullscreen" : "Expand to fullscreen"}
                onClick={() => setFullscreen((value) => !value)}
              />
            )}
          </div>
        </header>
      )}

      <div className="flex flex-1 flex-col px-5 py-4">{body}</div>

      {legend && !loading && !empty && (
        <div className="neer-divider flex flex-wrap items-center gap-4 px-5 py-3">{legend}</div>
      )}
    </div>
  );

  if (!allowFullscreen) return content;

  return (
    <AnimatePresence>
      {fullscreen ? (
        <>
          <motion.div
            className="fixed inset-0 z-modal bg-bg-base/90 backdrop-blur-sm"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={scaleIn}
            onClick={() => setFullscreen(false)}
            aria-hidden="true"
          />
          <motion.div initial="hidden" animate="visible" exit="exit" variants={scaleIn} className="contents">
            {content}
          </motion.div>
        </>
      ) : (
        content
      )}
    </AnimatePresence>
  );
}