"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Button
//
// Reusable action button. Purely presentational/interactive — no
// page-specific logic. Compose it with an onClick handler from the caller.
// -----------------------------------------------------------------------------

import { forwardRef } from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { DURATION, EASE_OUT, tapPress } from "@/lib/motion";
import { ICON_SIZES } from "@/lib/icons";

const VARIANT_CLASSES = {
  primary:
    "bg-accent-500 text-text-inverse border border-transparent shadow-glow-sm " +
    "hover:bg-accent-400 hover:shadow-glow active:bg-accent-600",
  secondary:
    "bg-surface-raised text-text-primary border border-border-strong " +
    "hover:bg-surface-overlay active:bg-surface",
  ghost:
    "bg-transparent text-text-secondary border border-transparent " +
    "hover:bg-surface-raised hover:text-text-primary active:bg-surface",
  // QA (Phase 31E): no custom glow here — glow is reserved for the primary
  // accent action per the "minimal glow" visual spec; danger relies on
  // color/contrast alone, which also reads calmer for a scientific app.
  danger: "bg-error text-text-inverse border border-transparent hover:bg-error/90 active:bg-error/80",
};

const SIZE_CLASSES = {
  sm: "h-8 px-3 text-small gap-1.5",
  md: "h-10 px-4 text-body gap-2",
  lg: "h-12 px-5 text-body gap-2.5",
};

const ICON_ONLY_SIZE_CLASSES = {
  sm: "h-8 w-8 p-0",
  md: "h-10 w-10 p-0",
  lg: "h-12 w-12 p-0",
};

const ICON_SIZE_BY_BUTTON_SIZE = {
  sm: "sm",
  md: "sm",
  lg: "lg",
};

/**
 * @param {"primary"|"secondary"|"ghost"|"danger"} variant
 * @param {"sm"|"md"|"lg"} size
 * @param {React.ComponentType} icon - a lucide-react icon component
 * @param {"left"|"right"} iconPosition
 * @param {boolean} iconOnly - renders a square icon-only button; requires aria-label
 * @param {boolean} loading - shows a spinner and disables interaction
 * @param {boolean} disabled
 */
const Button = forwardRef(function Button(
  {
    variant = "primary",
    size = "md",
    icon: Icon,
    iconPosition = "left",
    iconOnly = false,
    loading = false,
    disabled = false,
    className,
    children,
    type = "button",
    "aria-label": ariaLabel,
    ...props
  },
  ref
) {
  const isDisabled = disabled || loading;
  const iconSizeKey = ICON_SIZE_BY_BUTTON_SIZE[size] ?? "sm";

  if (process.env.NODE_ENV !== "production" && iconOnly && !ariaLabel) {
    // eslint-disable-next-line no-console
    console.warn("Button: icon-only buttons require an aria-label for accessibility.");
  }

  return (
    <motion.button
      ref={ref}
      type={type}
      disabled={isDisabled}
      aria-busy={loading || undefined}
      aria-label={ariaLabel}
      whileHover={!isDisabled ? { y: -1 } : undefined}
      whileTap={!isDisabled ? tapPress : undefined}
      transition={{ duration: DURATION.fast, ease: EASE_OUT }}
      className={cn(
        "inline-flex select-none items-center justify-center rounded-md font-medium neer-transition",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
        "focus-visible:ring-offset-2 focus-visible:ring-offset-bg-base",
        VARIANT_CLASSES[variant] ?? VARIANT_CLASSES.primary,
        iconOnly ? ICON_ONLY_SIZE_CLASSES[size] : SIZE_CLASSES[size],
        isDisabled && "pointer-events-none opacity-45",
        className
      )}
      {...props}
    >
      {loading && (
        <Loader2
          size={ICON_SIZES[iconSizeKey]}
          strokeWidth={2}
          className="animate-spin"
          aria-hidden="true"
        />
      )}
      {!loading && Icon && iconPosition === "left" && (
        <Icon size={ICON_SIZES[iconSizeKey]} strokeWidth={1.75} aria-hidden="true" />
      )}
      {!iconOnly && children && <span>{children}</span>}
      {!loading && Icon && iconPosition === "right" && !iconOnly && (
        <Icon size={ICON_SIZES[iconSizeKey]} strokeWidth={1.75} aria-hidden="true" />
      )}
    </motion.button>
  );
});

export default Button;