"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Drawer
//
// Edge-anchored panel (left/right/top/bottom) rendered into a portal. Shares
// focus-trap/escape/scroll-lock behavior with Modal via useDialog.
// -----------------------------------------------------------------------------

import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";
import { fade, DURATION, EASE_OUT } from "@/lib/motion";
import { useDialog } from "@/lib/useDialog";
import Button from "../primitive/Button";

const POSITION_CONFIG = {
  right: { side: "right-0 top-0 h-full", edge: "border-l", size: "w-full max-w-sm", axis: "x", offset: 32 },
  left: { side: "left-0 top-0 h-full", edge: "border-r", size: "w-full max-w-sm", axis: "x", offset: -32 },
  top: { side: "top-0 left-0 w-full", edge: "border-b", size: "max-h-[80vh]", axis: "y", offset: -32 },
  bottom: { side: "bottom-0 left-0 w-full", edge: "border-t", size: "max-h-[80vh]", axis: "y", offset: 32 },
};

function buildSlideVariants(config) {
  const hidden = config.axis === "x" ? { x: config.offset } : { y: config.offset };
  const visible = config.axis === "x" ? { x: 0 } : { y: 0 };
  return {
    hidden: { opacity: 0, ...hidden },
    visible: { opacity: 1, ...visible, transition: { duration: DURATION.slow, ease: EASE_OUT } },
    exit: { opacity: 0, ...hidden, transition: { duration: DURATION.fast, ease: EASE_OUT } },
  };
}

/**
 * @param {boolean} open
 * @param {() => void} onClose
 * @param {"left"|"right"|"top"|"bottom"} position
 * @param {string} title
 * @param {string} subtitle - optional line rendered under the title
 * @param {React.ReactNode} headerActions - optional controls rendered before the close button
 * @param {boolean} showClose - set false to omit the built-in close button (e.g. a caller renders its own)
 * @param {React.ReactNode} footer - optional content pinned below the scrollable body
 * @param {React.ReactNode} children
 */
export default function Drawer({
  open,
  onClose,
  position = "right",
  title,
  subtitle,
  headerActions,
  showClose = true,
  footer,
  children,
  className,
  bodyClassName,
}) {
  const containerRef = useDialog({ open, onClose });
  const config = POSITION_CONFIG[position] ?? POSITION_CONFIG.right;
  const variants = buildSlideVariants(config);
  const hasHeader = Boolean(title || subtitle || headerActions || showClose);

  if (typeof document === "undefined") return null;

  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-modal">
          <motion.div
            className="absolute inset-0 bg-bg-base/80 backdrop-blur-sm"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={fade}
            onClick={onClose}
            aria-hidden="true"
          />

          <motion.div
            ref={containerRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={title ? "neer-drawer-title" : undefined}
            tabIndex={-1}
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={variants}
            className={cn(
              "neer-panel-raised absolute flex flex-col",
              config.side,
              config.edge,
              config.size,
              className
            )}
          >
            {hasHeader && (
              <header className="neer-divider flex shrink-0 items-start justify-between gap-4 px-5 py-4">
                <div className="min-w-0">
                  {title ? (
                    <h3 id="neer-drawer-title" className="truncate text-h3 text-text-primary">
                      {title}
                    </h3>
                  ) : (
                    <span />
                  )}
                  {subtitle && <p className="mt-0.5 truncate text-small text-text-muted">{subtitle}</p>}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {headerActions}
                  {showClose && (
                    <Button variant="ghost" size="sm" iconOnly icon={X} aria-label="Close" onClick={onClose} />
                  )}
                </div>
              </header>
            )}

            <div className={cn("flex-1 overflow-y-auto px-5 py-4", bodyClassName)}>{children}</div>

            {footer && <div className="neer-divider shrink-0 bg-surface-sunken/40 px-5 py-3">{footer}</div>}
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body
  );
}