"use client";

// -----------------------------------------------------------------------------
// NEER Design System — ErrorState
//
// A calm, professional error presentation for failed data loads/requests —
// not an alarming crash screen. Optional retry action and collapsible
// technical details for debugging without cluttering the primary message.
// -----------------------------------------------------------------------------

import { motion } from "framer-motion";
import { AlertTriangle, RotateCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { fade } from "@/lib/motion";
import Button from "../primitives/Button";

/**
 * @param {string} title
 * @param {string} message
 * @param {React.ComponentType} icon - defaults to AlertTriangle
 * @param {() => void} onRetry - if provided, shows a "Retry" button
 * @param {string} retryLabel
 * @param {string} details - optional technical details (stack trace, error code, etc.)
 */
export default function ErrorState({
  title = "Something went wrong",
  message = "This data couldn't be loaded. Please try again.",
  icon: Icon = AlertTriangle,
  onRetry,
  retryLabel = "Retry",
  details,
  className,
}) {
  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fade}
      role="alert"
      className={cn(
        "flex flex-col items-center gap-3 rounded-md border border-error-border bg-error-bg px-6 py-8 text-center",
        className
      )}
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-full border border-error-border bg-surface-base text-error">
        <Icon size={20} strokeWidth={1.75} aria-hidden="true" />
      </span>

      <div className="space-y-1">
        <h4 className="text-h3 text-text-primary">{title}</h4>
        <p className="max-w-sm text-small text-text-muted">{message}</p>
      </div>

      {onRetry && (
        <Button variant="secondary" size="sm" icon={RotateCw} onClick={onRetry}>
          {retryLabel}
        </Button>
      )}

      {details && (
        <details className="mt-2 w-full max-w-md text-left">
          <summary className="cursor-pointer text-caption text-text-muted hover:text-text-secondary">
            Technical details
          </summary>
          <pre className="mt-2 max-h-40 overflow-auto rounded-md border border-border bg-surface-sunken/60 p-3 text-caption text-text-muted">
            {details}
          </pre>
        </details>
      )}
    </motion.div>
  );
}