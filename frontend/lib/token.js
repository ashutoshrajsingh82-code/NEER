// -----------------------------------------------------------------------------
// NEER Design System — raw token values for JS contexts
//
// Tailwind utilities (bg-*, text-*, border-*) and the CSS variables in
// app/globals.css are the source of truth for styling markup. This file
// mirrors those same values as plain JS so charting/mapping libraries that
// can't read Tailwind classes (Chart.js, D3, deck.gl, canvas draws, SVG
// generated on the server) stay visually consistent with the rest of the UI.
//
// Keep this file in sync with tailwind.config.js if tokens change.
// -----------------------------------------------------------------------------

export const colors = {
  bg: {
    base: "#050B14",
    deep: "#071322",
    DEFAULT: "#0A1626",
    raised: "#0E1F33",
  },
  surface: {
    base: "#0C1B2E",
    raised: "#112238",
    overlay: "#152A44",
    sunken: "#081420",
  },
  border: {
    subtle: "rgba(148, 197, 224, 0.08)",
    DEFAULT: "rgba(148, 197, 224, 0.14)",
    strong: "rgba(148, 197, 224, 0.24)",
    accent: "rgba(45, 212, 191, 0.45)",
  },
  accent: {
    50: "#ECFEFF",
    100: "#CFFAFE",
    200: "#A5F3FC",
    300: "#67E8F9",
    400: "#2DD4BF",
    500: "#14B8A6",
    600: "#0D9488",
    700: "#0F766E",
    800: "#115E59",
    900: "#134E4A",
    DEFAULT: "#22D3EE",
    glow: "#67E8F9",
  },
  text: {
    primary: "#E6F1F8",
    secondary: "#AEC5D6",
    muted: "#6B839A",
    disabled: "#41576B",
    inverse: "#04121D",
  },
  success: "#34D399",
  warning: "#FBBF24",
  error: "#F87171",
  info: "#38BDF8",
};

/** Ordered categorical palette for multi-series charts (depth profiles, etc). */
export const chartCategoricalPalette = [
  colors.accent.DEFAULT, // cyan
  colors.accent[400], // teal
  colors.info, // sky blue
  colors.warning, // amber
  colors.success, // emerald
  colors.error, // coral red
  colors.accent[200], // pale cyan
  colors.accent[700], // deep teal
];

/** Sequential palette for depth/heatmap-style continuous data, dark → bright cyan. */
export const chartSequentialPalette = [
  colors.bg.raised,
  colors.accent[900],
  colors.accent[700],
  colors.accent[500],
  colors.accent[400],
  colors.accent[300],
  colors.accent[100],
];

export const radius = {
  sm: 4,
  DEFAULT: 8,
  md: 10,
  lg: 14,
  xl: 20,
  "2xl": 28,
};

export const spacing = {
  1: 4,
  2: 8,
  3: 12,
  4: 16,
  5: 20,
  6: 24,
  8: 32,
  10: 40,
  12: 48,
  16: 64,
};

export const typography = {
  fontSans:
    '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  fontMono:
    '"JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace',
  sizes: {
    display: 56,
    h1: 40,
    h2: 30,
    h3: 22,
    body: 16,
    small: 14,
    caption: 12,
    data: 14,
  },
};

export const shadows = {
  panel: "0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px -8px rgba(0, 0, 0, 0.5)",
  raised: "0 4px 12px rgba(0, 0, 0, 0.45), 0 16px 40px -16px rgba(0, 0, 0, 0.6)",
  glowSm: "0 0 12px rgba(34, 211, 238, 0.18)",
  glow: "0 0 24px rgba(34, 211, 238, 0.22)",
  glowLg: "0 0 48px rgba(34, 211, 238, 0.28)",
};

export const transitions = {
  duration: { fast: 120, base: 200, slow: 320, slower: 480 },
  easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
};

const tokens = { colors, chartCategoricalPalette, chartSequentialPalette, radius, spacing, typography, shadows, transitions };

export default tokens;