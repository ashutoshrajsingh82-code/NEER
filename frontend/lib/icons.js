// -----------------------------------------------------------------------------
// NEER Design System — Icon defaults (Lucide React)
//
// Import icon components directly from "lucide-react" as usual, and spread
// `iconProps(...)` onto them so size/stroke-width/color stay consistent
// across the app instead of being set ad hoc on every usage.
//
//   import { Waves } from "lucide-react";
//   import { iconProps } from "@/lib/icons";
//
//   <Waves {...iconProps()} />
//   <Waves {...iconProps("lg", "text-accent")} />
// -----------------------------------------------------------------------------

/** Icon size scale, in pixels — mirrors the type scale's rhythm. */
export const ICON_SIZES = {
  xs: 14,
  sm: 16,
  md: 18, // default — pairs with body/small text
  lg: 22,
  xl: 28,
  "2xl": 36,
};

/** Default stroke width used across the app for a crisp, technical line weight. */
export const DEFAULT_STROKE_WIDTH = 1.75;

/**
 * Returns the standard props to spread onto any lucide-react icon component.
 * @param {keyof typeof ICON_SIZES} size
 * @param {string} className - Tailwind classes, e.g. "text-accent" for color.
 * @param {number} strokeWidth
 */
export function iconProps(size = "md", className = "text-text-secondary", strokeWidth = DEFAULT_STROKE_WIDTH) {
  return {
    size: ICON_SIZES[size] ?? ICON_SIZES.md,
    strokeWidth,
    className,
  };
}