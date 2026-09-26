// -----------------------------------------------------------------------------
// NEER Design System — Framer Motion defaults
//
// Centralized, subtle motion primitives so every component animates with the
// same easing/duration language instead of one-off values. Motion here is
// intentionally restrained (small offsets, quick durations) to match the
// "scientific instrumentation" feel rather than a playful/bouncy UI.
// -----------------------------------------------------------------------------

/** Standard easing curve used across the app (matches tailwind.config.js). */
export const EASE_OUT = [0.16, 1, 0.3, 1];

/** Shared duration scale, in seconds. */
export const DURATION = {
  fast: 0.12,
  base: 0.2,
  slow: 0.32,
  slower: 0.48,
};

/** Default spring used for small interactive elements (buttons, toggles). */
export const springSnappy = {
  type: "spring",
  stiffness: 420,
  damping: 32,
  mass: 0.7,
};

/** Softer spring for panels/cards entering the viewport. */
export const springSoft = {
  type: "spring",
  stiffness: 220,
  damping: 26,
  mass: 0.9,
};

/** Fade in + rise slightly — the default "content appears" transition. */
export const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: DURATION.slow, ease: EASE_OUT },
  },
  exit: {
    opacity: 0,
    y: -8,
    transition: { duration: DURATION.fast, ease: EASE_OUT },
  },
};

/** Plain fade, no movement — for backdrops, overlays, subtle swaps. */
export const fade = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: DURATION.base, ease: EASE_OUT } },
  exit: { opacity: 0, transition: { duration: DURATION.fast, ease: EASE_OUT } },
};

/** Scale + fade — for modals, popovers, dropdown menus. */
export const scaleIn = {
  hidden: { opacity: 0, scale: 0.96 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: DURATION.base, ease: EASE_OUT },
  },
  exit: {
    opacity: 0,
    scale: 0.98,
    transition: { duration: DURATION.fast, ease: EASE_OUT },
  },
};

/** Slide in from the side — for drawers/side panels. */
export const slideInRight = {
  hidden: { opacity: 0, x: 24 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: DURATION.slow, ease: EASE_OUT },
  },
  exit: { opacity: 0, x: 16, transition: { duration: DURATION.fast, ease: EASE_OUT } },
};

/**
 * Stagger container — wrap a list of `fadeUp` children in a motion.div with
 * `variants={staggerContainer}` to have them animate in sequence.
 */
export const staggerContainer = (staggerChildren = 0.06, delayChildren = 0) => ({
  hidden: {},
  visible: {
    transition: { staggerChildren, delayChildren },
  },
});

/** Subtle hover lift for cards/panels — spread onto `whileHover`. */
export const hoverLift = { y: -2, transition: { duration: DURATION.fast, ease: EASE_OUT } };

/** Subtle press feedback — spread onto `whileTap`. */
export const tapPress = { scale: 0.98, transition: { duration: DURATION.fast } };

/** Gentle ambient pulse, useful for live/status indicators. */
export const pulseGlow = {
  animate: {
    opacity: [0.6, 1, 0.6],
    transition: { duration: 2.4, repeat: Infinity, ease: "easeInOut" },
  },
};