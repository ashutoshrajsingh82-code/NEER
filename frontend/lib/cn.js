// -----------------------------------------------------------------------------
// NEER Design System — className helper
//
// Tiny `clsx`-style utility so components can conditionally join Tailwind
// classes without pulling in an extra dependency.
//
//   cn("neer-panel", isActive && "neer-glow-ring", className)
// -----------------------------------------------------------------------------

export function cn(...classes) {
  return classes.filter(Boolean).join(" ");
}

export default cn;