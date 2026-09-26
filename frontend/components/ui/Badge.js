// -----------------------------------------------------------------------------
// NEER Design System — Badge
//
// Small pill label for statuses/tags. Purely presentational.
// Built on top of the .neer-badge* utility classes in globals.css.
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";

const VARIANT_CLASSES = {
  default: "neer-badge bg-surface-raised border-border-strong text-text-secondary",
  success: "neer-badge-success",
  warning: "neer-badge-warning",
  error: "neer-badge-error",
  info: "neer-badge-info",
  // Scientific/data-status flavors — for instrument/reading states rather
  // than generic UI feedback (e.g. "CALIBRATED", "NOMINAL", "NO SIGNAL").
  accent: "neer-badge-accent",
  neutral: "neer-badge bg-surface-sunken border-border text-text-muted",
};

const SIZE_CLASSES = {
  sm: "text-[10px] px-2 py-0.5 gap-1",
  md: "",
};

/**
 * @param {"default"|"success"|"warning"|"error"|"info"|"accent"|"neutral"} variant
 * @param {React.ComponentType} icon - optional lucide-react icon
 * @param {"sm"|"md"} size
 */
export default function Badge({
  variant = "default",
  icon: Icon,
  size = "md",
  className,
  children,
  ...props
}) {
  return (
    <span
      className={cn(
        VARIANT_CLASSES[variant] ?? VARIANT_CLASSES.default,
        SIZE_CLASSES[size],
        className
      )}
      {...props}
    >
      {Icon && <Icon size={12} strokeWidth={2} aria-hidden="true" />}
      {children}
    </span>
  );
}