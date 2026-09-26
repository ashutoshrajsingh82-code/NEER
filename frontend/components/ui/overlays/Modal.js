"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Modal
//
// Centered dialog rendered into a portal. Focus management, escape-to-close,
// and scroll locking are handled by the shared useDialog hook so Modal and
// Drawer stay consistent. Controlled by the caller via `open`/`onClose`.
// -----------------------------------------------------------------------------

import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";
import { fade, scaleIn } from "@/lib/motion";
import { useDialog } from "@/lib/useDialog";
import Button from "../primitives/Button";

const SIZE_CLASSES = {
  sm: "max-w-sm",
  md: "max-w-lg",
  lg: "max-w-2xl",
};

/**
 * @param {boolean} open
 * @param {() => void} onClose - also called on Escape and backdrop click
 * @param {string} title
 * @param {string} description
 * @param {React.ReactNode} children - modal body content
 * @param {React.ReactNode} footer - typically action buttons
 * @param {"sm"|"md"|"lg"} size
 */
export default function Modal({ open, onClose, title, description, children, footer, size = "md", className }) {
  const containerRef = useDialog({ open, onClose });

  if (typeof document === "undefined") return null;

  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-modal flex items-center justify-center p-4">
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
            aria-labelledby={title ? "neer-modal-title" : undefined}
            aria-describedby={description ? "neer-modal-description" : undefined}
            tabIndex={-1}
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={scaleIn}
            className={cn(
              "neer-panel-raised relative z-modal flex max-h-[85vh] w-full flex-col",
              SIZE_CLASSES[size] ?? SIZE_CLASSES.md,
              className
            )}
          >
            {(title || description || onClose) && (
              <header className="neer-divider flex items-start justify-between gap-4 px-5 py-4">
                <div className="min-w-0">
                  {title && (
                    <h3 id="neer-modal-title" className="text-h3 text-text-primary">
                      {title}
                    </h3>
                  )}
                  {description && (
                    <p id="neer-modal-description" className="mt-1 text-small text-text-muted">
                      {description}
                    </p>
                  )}
                </div>
                {onClose && (
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={X}
                    aria-label="Close dialog"
                    onClick={onClose}
                  />
                )}
              </header>
            )}

            <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>

            {footer && (
              <footer className="neer-divider flex items-center justify-end gap-2 px-5 py-3">{footer}</footer>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body
  );
}