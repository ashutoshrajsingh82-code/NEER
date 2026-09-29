// -----------------------------------------------------------------------------
// NEER Application Shell — MainContent
//
// Owns the actual scrolling behavior of the primary content area (via
// overflow-y-auto) so the shell's other regions stay fixed in place. Semantic
// <main> element. No page-specific logic — pages render whatever they need
// as children.
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";

export default function MainContent({ id, children, className, ...props }) {
  return (
    <main
      id={id}
      // tabIndex(-1) lets a skip link (see AppShell's route-group layout)
      // move focus here directly, even though <main> isn't natively focusable.
      tabIndex={-1}
      className={cn(
        "min-w-0 flex-1 overflow-y-auto bg-grid-subtle bg-grid px-3 py-4 focus:outline-none sm:px-6 sm:py-6",
        className
      )}
      {...props}
    >
      {children}
    </main>
  );
}
