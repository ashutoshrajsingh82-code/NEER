/** @type {import('tailwindcss').Config} */

// ---------------------------------------------------------------------------
// NEER Design System — Tailwind configuration
//
// Visual language: dark navy/ocean "scientific control-center" aesthetic.
// Deep navy backgrounds, layered glass-like surfaces, thin hairline borders,
// cyan/teal accents, restrained glow, and a clear typographic hierarchy for
// dense data-heavy interfaces (charts, maps, reconstruction panels).
//
// Every token below is exposed as a Tailwind utility (bg-bg-base,
// text-text-primary, border-border-subtle, shadow-glow-sm, etc.) so
// components never hardcode raw hex values.
// ---------------------------------------------------------------------------

module.exports = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
    "./lib/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Background layers — deepest to shallowest
        bg: {
          base: "#050B14", // page background — near-black ocean navy
          deep: "#071322", // deepest panels / sidebars
          DEFAULT: "#0A1626", // primary app background
          raised: "#0E1F33", // raised sections
        },
        // Surface / panel colors — cards, panels, modals
        surface: {
          base: "#0C1B2E", // default panel
          raised: "#112238", // elevated panel / hover
          overlay: "#152A44", // popovers, dropdowns, tooltips
          sunken: "#081420", // inset / well surfaces
        },
        // Border colors — thin hairlines throughout
        border: {
          subtle: "rgba(148, 197, 224, 0.08)", // default hairline
          DEFAULT: "rgba(148, 197, 224, 0.14)", // standard divider
          strong: "rgba(148, 197, 224, 0.24)", // emphasized divider
          accent: "rgba(45, 212, 191, 0.45)", // cyan-tinted border (focus/active)
        },
        // Primary cyan / teal accent scale
        accent: {
          50: "#ECFEFF",
          100: "#CFFAFE",
          200: "#A5F3FC",
          300: "#67E8F9",
          400: "#2DD4BF", // core teal
          500: "#14B8A6", // primary accent
          600: "#0D9488",
          700: "#0F766E",
          800: "#115E59",
          900: "#134E4A",
          DEFAULT: "#22D3EE", // primary cyan
          glow: "#67E8F9",
        },
        // Text colors
        text: {
          primary: "#E6F1F8", // headings, high-emphasis
          secondary: "#AEC5D6", // body copy
          muted: "#6B839A", // captions, meta, placeholders
          disabled: "#41576B",
          inverse: "#04121D", // text on light/accent surfaces
        },
        // Semantic state colors
        success: {
          DEFAULT: "#34D399",
          bg: "rgba(52, 211, 153, 0.12)",
          border: "rgba(52, 211, 153, 0.35)",
        },
        warning: {
          DEFAULT: "#FBBF24",
          bg: "rgba(251, 191, 36, 0.12)",
          border: "rgba(251, 191, 36, 0.35)",
        },
        error: {
          DEFAULT: "#F87171",
          bg: "rgba(248, 113, 113, 0.12)",
          border: "rgba(248, 113, 113, 0.35)",
        },
        info: {
          DEFAULT: "#38BDF8",
          bg: "rgba(56, 189, 248, 0.12)",
          border: "rgba(56, 189, 248, 0.35)",
        },
      },

      fontFamily: {
        sans: [
          "var(--font-sans)",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Inter",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "var(--font-mono)",
          "JetBrains Mono",
          "Fira Code",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },

      // Typography hierarchy — used via className, e.g. `text-display`, `text-h1`
      fontSize: {
        display: ["3.5rem", { lineHeight: "1.05", letterSpacing: "-0.02em", fontWeight: "700" }],
        h1: ["2.5rem", { lineHeight: "1.1", letterSpacing: "-0.015em", fontWeight: "700" }],
        h2: ["1.875rem", { lineHeight: "1.2", letterSpacing: "-0.01em", fontWeight: "600" }],
        h3: ["1.375rem", { lineHeight: "1.3", letterSpacing: "-0.005em", fontWeight: "600" }],
        body: ["1rem", { lineHeight: "1.6", fontWeight: "400" }],
        small: ["0.875rem", { lineHeight: "1.5", fontWeight: "400" }],
        caption: ["0.75rem", { lineHeight: "1.4", fontWeight: "500", letterSpacing: "0.02em" }],
        data: ["0.875rem", { lineHeight: "1.5", fontWeight: "500" }], // monospace/data text size
      },

      borderRadius: {
        none: "0",
        sm: "0.25rem", // 4px
        DEFAULT: "0.5rem", // 8px
        md: "0.625rem", // 10px
        lg: "0.875rem", // 14px
        xl: "1.25rem", // 20px
        "2xl": "1.75rem", // 28px
        full: "9999px",
      },

      spacing: {
        px: "1px",
        0.5: "0.125rem",
        1: "0.25rem",
        1.5: "0.375rem",
        2: "0.5rem",
        3: "0.75rem",
        4: "1rem",
        5: "1.25rem",
        6: "1.5rem",
        8: "2rem",
        10: "2.5rem",
        12: "3rem",
        16: "4rem",
        20: "5rem",
        24: "6rem",
        32: "8rem",
      },

      // Shadows — from a soft ambient panel shadow to a scientific accent glow
      boxShadow: {
        panel: "0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px -8px rgba(0, 0, 0, 0.5)",
        raised: "0 4px 12px rgba(0, 0, 0, 0.45), 0 16px 40px -16px rgba(0, 0, 0, 0.6)",
        "glow-sm": "0 0 12px rgba(34, 211, 238, 0.18)",
        glow: "0 0 24px rgba(34, 211, 238, 0.22)",
        "glow-lg": "0 0 48px rgba(34, 211, 238, 0.28)",
        "inner-line": "inset 0 1px 0 rgba(148, 197, 224, 0.06)",
      },

      // Subtle scientific grid + radial glow backgrounds
      backgroundImage: {
        "grid-subtle":
          "linear-gradient(rgba(148, 197, 224, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(148, 197, 224, 0.05) 1px, transparent 1px)",
        "radial-glow":
          "radial-gradient(60% 60% at 50% 0%, rgba(45, 212, 191, 0.12) 0%, rgba(5, 11, 20, 0) 70%)",
        "ocean-gradient": "linear-gradient(180deg, #071322 0%, #050B14 100%)",
      },
      backgroundSize: {
        grid: "32px 32px",
        "grid-lg": "64px 64px",
      },

      transitionDuration: {
        fast: "120ms",
        DEFAULT: "200ms",
        slow: "320ms",
        slower: "480ms",
      },
      transitionTimingFunction: {
        DEFAULT: "cubic-bezier(0.4, 0, 0.2, 1)",
        out: "cubic-bezier(0.16, 1, 0.3, 1)",
      },

      letterSpacing: {
        tightest: "-0.02em",
        widest: "0.08em",
      },

      zIndex: {
        base: "0",
        raised: "10",
        overlay: "40",
        modal: "50",
        toast: "60",
      },
    },
  },
  plugins: [],
};