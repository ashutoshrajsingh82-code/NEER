// -----------------------------------------------------------------------------
// NEER Application Shell — MainContent
//
// Owns the actual scrolling behavior of the primary content area (via
// overflow-y-auto) so the shell's other regions stay fixed in place. Semantic
// <main> element. No page-specific logic — pages render whatever they need
// as children.
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";

export default function MainContent({ children, className }) {
  return (
    <main className={cn("flex-1 overflow-y-auto bg-grid-subtle bg-grid px-6 py-6", className)}>
      {children}
    </main>
  );
}